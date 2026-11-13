"""LanceDB 向量存储 — 嵌入式的语义检索引擎"""

import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import pyarrow as pa
import lancedb

from idb.config import get_config


class LanceDBStore:
    """封装 LanceDB 操作，提供文件索引和语义搜索能力。

    LanceDB 是嵌入式向量数据库，无需服务端进程。
    数据存在 db_path 目录下，复制目录即备份。
    """

    def __init__(self, db_path: Optional[str] = None):
        cfg = get_config()
        self.db_path = db_path or cfg.db_path
        self.vectors_path = os.path.join(self.db_path, "vectors")
        self._db = None
        self._chunks_table = None
        self._files_table = None

    # ─── 初始化 ─────────────────────────────────────

    @property
    def db(self):
        if self._db is None:
            os.makedirs(self.vectors_path, exist_ok=True)
            self._db = lancedb.connect(self.vectors_path)
        return self._db

    @property
    def chunks_table(self):
        if self._chunks_table is None:
            try:
                self._chunks_table = self.db.open_table("chunks")
            except Exception:
                pass  # 将在首次 add_chunks 时创建
        return self._chunks_table

    @property
    def files_table(self):
        if self._files_table is None:
            self._files_table = self._get_or_create_files_table()
        return self._files_table

    def _get_or_create_chunks_table(self):
        """获取或创建 chunks 表（首次插入数据时自动推断 schema）"""
        table_name = "chunks"
        try:
            return self.db.open_table(table_name)
        except Exception:
            # 不预设 schema — 首次 add_chunks 时自动推断
            # LanceDB 会从 PyArrow record batch 中自动检测 vector 列类型
            return None

    def _ensure_chunks_table(self, sample_row: dict):
        """确保 chunks 表存在，若不存在则以样本行创建"""
        if self._chunks_table is not None:
            return
        table_name = "chunks"
        try:
            self._chunks_table = self.db.open_table(table_name)
        except Exception:
            # 从样本行推断 schema 并创建空表
            import pyarrow as pa
            keys = list(sample_row.keys())
            types = []
            for k in keys:
                v = sample_row[k]
                if k == "vector":
                    types.append(pa.list_(pa.float32(), len(v)))
                elif isinstance(v, str):
                    types.append(pa.string())
                elif isinstance(v, int):
                    types.append(pa.int32())
                else:
                    types.append(pa.string())
            schema = pa.schema([pa.field(k, t) for k, t in zip(keys, types)])
            self._chunks_table = self.db.create_table(table_name, schema=schema)

    def _get_or_create_files_table(self):
        """获取或创建 files 表（文件级索引）"""
        table_name = "files"
        try:
            return self.db.open_table(table_name)
        except Exception:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("path", pa.string()),
                pa.field("name", pa.string()),
                pa.field("hash", pa.string()),
                pa.field("mime_type", pa.string()),
                pa.field("size_bytes", pa.int64()),
                pa.field("chunk_count", pa.int32()),
                pa.field("ingested_at", pa.string()),
                pa.field("status", pa.string()),
            ])
            return self.db.create_table(table_name, schema=schema)

    # ─── 写入操作 ─────────────────────────────────

    def add_file_record(
        self,
        file_id: str,
        path: str,
        name: str,
        file_hash: str,
        mime_type: str,
        size_bytes: int,
        chunk_count: int,
    ):
        """记录一个文件的摄入"""
        self.files_table.add([{
            "id": file_id,
            "path": path,
            "name": name,
            "hash": file_hash,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "chunk_count": chunk_count,
            "ingested_at": datetime.now().isoformat(),
            "status": "active",
        }])

    def add_chunks(self, chunks: list, vectors: list[list[float]]):
        """批量写入 chunks 及其向量。

        Args:
            chunks: Chunk 对象列表
            vectors: 对应的向量列表
        """
        rows = []
        now = datetime.now().isoformat()
        for chunk, vec in zip(chunks, vectors):
            d = chunk.to_dict() if hasattr(chunk, 'to_dict') else chunk
            rows.append({
                "id": d["id"],
                "file_id": d["file_id"],
                "file_name": d["file_name"],
                "chunk_index": d["chunk_index"],
                "content": d["content"],
                "content_type": d.get("content_type", "text"),
                "token_count": d.get("token_count", 0),
                "location": d.get("location", ""),
                "vector": vec,
                "ingested_at": now,
            })

        # 确保表存在
        if self._chunks_table is None:
            self._ensure_chunks_table(rows[0])

        self._chunks_table.add(rows)

    def file_exists_by_hash(self, file_hash: str) -> bool:
        """检查相同哈希的文件是否已摄入"""
        try:
            result = self.files_table.search().where(
                f"hash = '{file_hash}' AND status = 'active'"
            ).limit(1).to_list()
            return len(result) > 0
        except Exception:
            return False

    def get_file_stats(self) -> dict:
        """获取数据库统计信息"""
        try:
            files = self.files_table.search().where("status = 'active'").to_list()
        except Exception:
            files = []
        try:
            chunks = self.chunks_table.search().limit(1).to_list()
            chunk_count = self.chunks_table.count_rows()
        except Exception:
            chunk_count = 0

        total_size = sum(f.get("size_bytes", 0) for f in files)
        return {
            "file_count": len(files),
            "chunk_count": chunk_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024*1024), 2),
        }

    # ─── 搜索操作 ─────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        *,
        where: Optional[str] = None,
    ) -> list[dict]:
        """向量语义搜索。

        Args:
            query_vector: 查询嵌入向量
            top_k: 返回结果数量
            where: 可选的 SQL 过滤条件

        Returns:
            匹配的 chunk 列表，包含 content, file_name, score 等
        """
        table = self.chunks_table
        if table is None:
            return []

        q = table.search(
            query_vector,
            vector_column_name="vector",
        ).limit(top_k)
        if where:
            q = q.where(where)
        results = q.to_list()

        # 格式化输出
        output = []
        for r in results:
            output.append({
                "id": r["id"],
                "file_name": r["file_name"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "content_type": r.get("content_type", "text"),
                "location": r.get("location", ""),
                "score": r.get("_distance", 0.0),  # cosine distance
            })
        return output
