"""SQLite 元数据存储 + FTS5 全文索引"""

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from idb.config import get_config


class SQLiteStore:
    """SQLite 数据库，存储文件/Chunk 元数据 + FTS5 全文搜索。

    与 LanceDB 互补：
    - LanceDB: 向量语义搜索
    - SQLite: 精确元数据过滤 + 全文关键词搜索 + 查询缓存
    """

    def __init__(self, db_path: Optional[str] = None):
        cfg = get_config()
        base = db_path or cfg.db_path
        self.db_file = os.path.join(base, "metadata.db")
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            self._conn = sqlite3.connect(self.db_file)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_tables()
        return self._conn

    def _init_tables(self):
        """初始化所有表"""
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            hash TEXT NOT NULL,
            mime_type TEXT,
            size_bytes INTEGER,
            chunk_count INTEGER,
            ingested_at TEXT,
            status TEXT DEFAULT 'active',
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            file_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            token_count INTEGER,
            location TEXT,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (file_id) REFERENCES files(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

        -- FTS5 全文索引：trigram tokenizer 支持中文无空格文本的子串匹配
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content, file_name, chunk_id,
            tokenize='trigram'
        );

        -- 触发器: 自动同步 FTS
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content, file_name, chunk_id)
            VALUES (new.rowid, new.content,
                    (SELECT name FROM files WHERE id = new.file_id),
                    new.id);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, content, file_name, chunk_id)
            VALUES ('delete', old.rowid, old.content,
                    (SELECT name FROM files WHERE id = old.file_id),
                    old.id);
        END;

        -- 查询缓存
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT,
            result_json TEXT,
            created_at TEXT,
            hit_count INTEGER DEFAULT 1
        );

        -- 摄入日志
        CREATE TABLE IF NOT EXISTS ingest_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            action TEXT,
            status TEXT,
            message TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        """)

    # ─── 文件操作 ───────────────────────────────

    def add_file(self, file_id: str, path: str, name: str, file_hash: str,
                 mime_type: str, size_bytes: int, chunk_count: int,
                 metadata: dict | None = None):
        self.conn.execute(
            """INSERT OR REPLACE INTO files (id, path, name, hash, mime_type,
               size_bytes, chunk_count, ingested_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, path, name, file_hash, mime_type, size_bytes,
             chunk_count, datetime.now().isoformat(), json.dumps(metadata or {}))
        )
        self.conn.commit()

    def file_exists_by_hash(self, file_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM files WHERE hash = ? AND status = 'active'", (file_hash,)
        ).fetchone()
        return row is not None

    def add_chunks(self, chunks: list):
        """批量写入 chunk 元数据"""
        rows = []
        for c in chunks:
            d = c.to_dict() if hasattr(c, 'to_dict') else c
            rows.append((
                d["id"], d["file_id"], d["chunk_index"], d["content"],
                d.get("content_type", "text"), d.get("token_count", 0),
                d.get("location", ""),
                json.dumps(d.get("metadata", {}), ensure_ascii=False)
            ))
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks (id, file_id, chunk_index, content,
               content_type, token_count, location, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        file_count = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE status='active'"
        ).fetchone()[0]
        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        total_size = self.conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) FROM files WHERE status='active'"
        ).fetchone()[0]
        return {
            "file_count": file_count,
            "chunk_count": chunk_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def list_files(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, chunk_count, size_bytes, mime_type, ingested_at "
            "FROM files WHERE status='active' ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_file(self, file_id: str):
        """软删除文件"""
        self.conn.execute(
            "UPDATE files SET status='deleted' WHERE id=?", (file_id,)
        )
        self.conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
        self.conn.commit()

    # ─── FTS5 全文搜索 ──────────────────────────

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 全文搜索（trigram 支持中文；短词补充 LIKE 匹配）"""
        fts_query = _to_fts_query(query)
        rows = []
        try:
            rows = self.conn.execute(
                """SELECT c.id, c.content, c.chunk_index, c.content_type,
                   c.location, f.name as file_name, c.file_id,
                   rank as score
                   FROM chunks_fts
                   JOIN chunks c ON chunks_fts.chunk_id = c.id
                   JOIN files f ON c.file_id = f.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit)
            ).fetchall()
        except Exception:
            pass

        results = [dict(r) for r in rows]
        seen = {r["id"] for r in results}

        # LIKE 补充：trigram 对 1-2 字中文失效，用 LIKE 兜底
        import re
        short_terms = [t for t in re.findall(r'[一-鿿]{1,2}', query) if t]
        if short_terms and len(results) < limit:
            like_rows = []
            for term in short_terms[:5]:
                try:
                    like_rows += self.conn.execute(
                        """SELECT c.id, c.content, c.chunk_index, c.content_type,
                           c.location, f.name as file_name, c.file_id, 0.5 as score
                           FROM chunks c JOIN files f ON c.file_id = f.id
                           WHERE f.status='active' AND c.content LIKE ?
                           LIMIT ?""",
                        (f'%{term}%', limit)
                    ).fetchall()
                except Exception:
                    pass
            for r in like_rows:
                rid = r["id"]
                if rid not in seen:
                    results.append(dict(r))
                    seen.add(rid)
                if len(results) >= limit:
                    break

        return results[:limit]

    def search_metadata(self, filters: dict, limit: int = 20) -> list[dict]:
        """基于元数据精确过滤（日期、类型等）"""
        conditions = []
        params = []
        if "date_from" in filters:
            conditions.append("f.ingested_at >= ?")
            params.append(filters["date_from"])
        if "date_to" in filters:
            conditions.append("f.ingested_at <= ?")
            params.append(filters["date_to"])
        if "mime_type" in filters:
            conditions.append("f.mime_type = ?")
            params.append(filters["mime_type"])
        if "content_type" in filters:
            conditions.append("c.content_type = ?")
            params.append(filters["content_type"])

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.conn.execute(
            f"""SELECT c.id, c.content, c.chunk_index, c.content_type,
                c.location, f.name as file_name, c.file_id
                FROM chunks c JOIN files f ON c.file_id = f.id
                WHERE f.status='active' AND {where}
                LIMIT ?""",
            params + [limit]
        ).fetchall()
        return [dict(r) for r in rows]

    # ─── 查询缓存 ───────────────────────────────

    def _cache_key(self, query: str) -> str:
        """生成包含数据库状态的缓存键，新摄入自动失效旧缓存"""
        import hashlib
        fc = self.conn.execute("SELECT COUNT(*) FROM files WHERE status='active'").fetchone()[0]
        cc = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        raw = f"{query}|files={fc}|chunks={cc}"
        return hashlib.md5(raw.encode()).hexdigest()

    def cache_get(self, query: str) -> dict | None:
        h = self._cache_key(query)
        row = self.conn.execute(
            "SELECT result_json FROM query_cache WHERE query_hash = ?", (h,)
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = ?", (h,)
            )
            self.conn.commit()
            return json.loads(row[0])
        return None

    def cache_set(self, query: str, result: dict):
        h = self._cache_key(query)
        self.conn.execute(
            """INSERT OR REPLACE INTO query_cache
               (query_hash, query_text, result_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (h, query, json.dumps(result, ensure_ascii=False), datetime.now().isoformat())
        )
        self.conn.commit()

    def log_ingest(self, file_name: str, action: str, status: str, message: str = ""):
        self.conn.execute(
            "INSERT INTO ingest_log (file_name, action, status, message) VALUES (?, ?, ?, ?)",
            (file_name, action, status, message)
        )
        self.conn.commit()


def _to_fts_query(nl_query: str) -> str:
    """将自然语言查询转为 FTS5 trigram 兼容查询。

    trigram tokenizer 需要至少 3 个字符的 token 才能匹配。
    因此从自然语言查询中提取连续的中文/字母数字片段，
    只保留长度 >= 3 的片段，用 OR 连接。
    短词（1-2字）由 search_fts 的 LIKE 补充处理。
    """
    import re
    # 提取连续的中文/字母数字混合片段（保留 口腔CT / SBRT质控 这类混合token）
    tokens = re.findall(r'[一-鿿a-zA-Z0-9]+', nl_query)
    trigram_tokens = []
    for t in tokens:
        if len(t) >= 3:
            # 生成 3 字滑动窗口（覆盖子串匹配，包含中英混合）
            for i in range(max(1, len(t) - 2)):
                trigram_tokens.append(t[i:i+3])
    # 去重并限制数量
    seen = set()
    uniq = []
    for t in trigram_tokens[:20]:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return nl_query
    return " OR ".join(f'"{t}"' for t in uniq)
