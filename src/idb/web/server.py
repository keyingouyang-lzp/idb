"""FastAPI 后端 — 智能数据库 Web 服务"""

import os
import sys
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 确保 idb 在路径中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from idb.config import init_config, get_config
from idb.ingestion.parser import parse_file
from idb.ingestion.chunker import chunk_document
from idb.ingestion.embedder import embed_chunks
from idb.storage.lancedb_store import LanceDBStore
from idb.storage.sqlite_store import SQLiteStore
from idb.llm.client import LLMClient
from idb.query.retriever import retrieve
from idb.query.extractor import extract_and_structure

# ─── 初始化 ─────────────────────────────────

static_dir = Path(__file__).parent / "static"
app = FastAPI(title="idb — 智能数据库", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件（懒加载）
_lance: Optional[LanceDBStore] = None
_sqlite: Optional[SQLiteStore] = None
_llm: Optional[LLMClient] = None


def get_lance() -> LanceDBStore:
    global _lance
    if _lance is None:
        _lance = LanceDBStore()
    return _lance


def get_sqlite() -> SQLiteStore:
    global _sqlite
    if _sqlite is None:
        _sqlite = SQLiteStore()
    return _sqlite


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ─── API 路由 ────────────────────────────────


class AskRequest(BaseModel):
    query: str
    fields: Optional[list[str]] = None
    top_k: int = 20


@app.get("/api/stats")
async def api_stats():
    """数据库统计"""
    lance = get_lance()
    sqlite = get_sqlite()
    ls = lance.get_file_stats()
    ss = sqlite.get_stats()
    return {
        "files": sqlite.list_files(),
        "file_count": ss["file_count"],
        "chunk_count": ss["chunk_count"],
        "total_size_mb": ss["total_size_mb"],
    }


@app.post("/api/ingest")
async def api_ingest(file: UploadFile = File(...)):
    """上传并摄入文件"""
    lance = get_lance()
    sqlite = get_sqlite()
    llm = get_llm()

    # 保存临时文件
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: 解析
        parsed = parse_file(tmp_path)

        # Step 2: 去重
        if sqlite.file_exists_by_hash(parsed.hash):
            return JSONResponse(
                {"status": "skipped", "reason": "文件已存在", "name": file.filename},
                status_code=200,
            )

        # Step 3: 分块
        file_id = str(uuid.uuid4())
        chunks = chunk_document(file_id, parsed.file_name, parsed.text, parsed.tables)

        if not chunks:
            return JSONResponse(
                {"status": "empty", "reason": "无可索引内容", "name": file.filename},
                status_code=200,
            )

        # Step 4: 向量嵌入
        vectors = embed_chunks(chunks, client=llm)

        # Step 5: 双路写入
        sqlite.add_file(
            file_id, parsed.file_path, parsed.file_name, parsed.hash,
            parsed.mime_type, parsed.size_bytes, len(chunks),
        )
        sqlite.add_chunks(chunks)
        lance.add_file_record(
            file_id, parsed.file_path, parsed.file_name, parsed.hash,
            parsed.mime_type, parsed.size_bytes, len(chunks),
        )
        lance.add_chunks(chunks, vectors)

        sqlite.log_ingest(file.filename, "ingest", "success", f"{len(chunks)} chunks")

        return {
            "status": "ok",
            "name": file.filename,
            "chunks": len(chunks),
            "size_mb": round(parsed.size_bytes / (1024 * 1024), 2),
            "ocr_used": parsed.metadata.get("ocr_used", False),
        }

    except Exception as e:
        sqlite.log_ingest(file.filename, "ingest", "error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    """自然语言查询"""
    lance = get_lance()
    sqlite = get_sqlite()
    llm = get_llm()

    # 检查缓存
    cached = sqlite.cache_get(req.query)
    if cached and "records" in cached:
        return cached

    # 检索
    chunks = retrieve(
        query=req.query,
        store=lance,
        sqlite=sqlite,
        client=llm,
        top_k=req.top_k,
        use_cache=False,
    )

    if not chunks:
        return {
            "query": req.query,
            "fields": [],
            "records": [],
            "source_files": [],
            "avg_confidence": 0.0,
            "warnings": ["未找到相关文档。请尝试调整查询范围或关键词。"],
        }

    # 提取结构化
    cfg = get_config()
    result = extract_and_structure(
        user_query=req.query,
        chunks=chunks[:cfg.top_k_rerank],
        client=llm,
        target_fields=req.fields or None,
    )

    # 写缓存
    sqlite.cache_set(req.query, result)

    return result


@app.get("/api/search_simple")
async def api_search(q: str, top: int = 5):
    """快速搜索（不经过 LLM 提取，直接返回 chunks）"""
    lance = get_lance()
    llm = get_llm()

    query_vec = llm.embed_single(q)
    results = lance.search(query_vec, top_k=top)

    # 格式化为前端友好的格式
    return [
        {
            "file_name": r["file_name"],
            "content": r["content"][:300] + ("..." if len(r["content"]) > 300 else ""),
            "relevance": round(1.0 - min(r.get("score", 1.0), 2.0) / 2.0, 2),
            "location": r.get("location", ""),
        }
        for r in results
    ]


@app.delete("/api/files/{file_name:path}")
async def api_delete_file(file_name: str):
    """删除文件"""
    sqlite = get_sqlite()
    rows = sqlite.conn.execute(
        "SELECT id FROM files WHERE name = ? AND status = 'active'", (file_name,)
    ).fetchall()
    for row in rows:
        sqlite.delete_file(row[0])
    return {"status": "ok", "deleted": len(rows)}


class ExportRequest(BaseModel):
    query: str
    fields: Optional[list[str]] = None
    top_k: int = 30


@app.post("/api/export")
async def api_export(req: ExportRequest):
    """执行查询并导出为 Excel 文件"""
    import io
    import asyncio
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    lance = get_lance()
    sqlite = get_sqlite()
    llm = get_llm()

    # 在后台线程中执行阻塞的 LLM 调用，避免阻塞事件循环
    loop = asyncio.get_event_loop()

    # 检索（轻量，可同步）
    chunks = await loop.run_in_executor(None, lambda: retrieve(
        query=req.query, store=lance, sqlite=sqlite, client=llm,
        top_k=req.top_k, use_cache=False,
    ))

    if not chunks:
        raise HTTPException(status_code=404, detail="未找到相关文档")

    cfg = get_config()
    # LLM 提取（重量，线程池执行）
    result = await loop.run_in_executor(None, lambda: extract_and_structure(
        user_query=req.query, chunks=chunks[:cfg.top_k_rerank],
        client=llm, target_fields=req.fields or None,
    ))

    records = result.get("records", [])
    if not records:
        raise HTTPException(status_code=404, detail="未提取到结构化数据")

    # ── 构建 Excel ──
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "查询结果"

    # 提取字段
    fields = []
    for rec in records:
        for k in rec:
            if not k.startswith("_") and k not in fields:
                fields.append(k)

    # 样式
    header_font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="2B579A", end_color="2B579A", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center")
    cell_font = Font(name="Microsoft YaHei", size=10)
    cell_align = Alignment(vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin", color="CCCCCC"),
        right=Side(style="thin", color="CCCCCC"),
        top=Side(style="thin", color="CCCCCC"),
        bottom=Side(style="thin", color="CCCCCC"),
    )

    # 标题行
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(fields) + 2)
    title_cell = ws.cell(row=1, column=1, value=f"查询: {req.query[:80]}")
    title_cell.font = Font(name="Microsoft YaHei", bold=True, size=13, color="1A3C6E")
    title_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 30

    # 表头
    headers = fields + ["置信度", "来源"]
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=3, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border
    ws.row_dimensions[3].height = 22

    # 数据行
    for row_idx, rec in enumerate(records):
        excel_row = row_idx + 4
        for col_idx, field in enumerate(fields, 1):
            val = rec.get(field)
            if val is None:
                val = "-"
            elif isinstance(val, (dict, list)):
                val = str(val)[:200]
            cell = ws.cell(row=excel_row, column=col_idx, value=str(val))
            cell.font = cell_font
            cell.alignment = cell_align
            cell.border = thin_border

        # 置信度
        conf = rec.get("_confidence")
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else "-"
        conf_cell = ws.cell(row=excel_row, column=len(fields) + 1, value=conf_str)
        conf_cell.font = cell_font
        conf_cell.alignment = Alignment(horizontal="center", vertical="center")
        conf_cell.border = thin_border
        if isinstance(conf, (int, float)):
            if conf >= 0.8:
                conf_cell.fill = PatternFill(start_color="E8F5E9", end_color="E8F5E9", fill_type="solid")
            elif conf >= 0.6:
                conf_cell.fill = PatternFill(start_color="FFF8E1", end_color="FFF8E1", fill_type="solid")
            else:
                conf_cell.fill = PatternFill(start_color="FFEBEE", end_color="FFEBEE", fill_type="solid")

        # 来源
        src = rec.get("_source", "-")
        if isinstance(src, (dict, list)):
            src = str(src)[:200]
        src_cell = ws.cell(row=excel_row, column=len(fields) + 2, value=str(src)[:200])
        src_cell.font = Font(name="Microsoft YaHei", size=9, color="666666")
        src_cell.alignment = cell_align
        src_cell.border = thin_border

    # 列宽
    ws.column_dimensions['A'].width = 5
    for col_idx in range(1, len(fields) + 3):
        col_letter = openpyxl.utils.get_column_letter(col_idx)
        if col_idx <= len(fields):
            ws.column_dimensions[col_letter].width = max(18, min(40, len(str(headers[col_idx-1])) * 3))
        elif col_idx == len(fields) + 1:
            ws.column_dimensions[col_letter].width = 10
        else:
            ws.column_dimensions[col_letter].width = 40

    # 来源文件信息
    info_row = len(records) + 5
    ws.merge_cells(start_row=info_row, start_column=1, end_row=info_row, end_column=len(fields) + 2)
    info_cell = ws.cell(row=info_row, column=1,
                        value=f"来源文件: {', '.join(result.get('source_files', []))}  |  "
                              f"平均置信度: {result.get('avg_confidence', 0):.0%}  |  "
                              f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info_cell.font = Font(name="Microsoft YaHei", size=9, color="999999")

    # 写入 BytesIO
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)

    # 生成文件名
    safe_name = req.query[:20].replace("/", "_").replace(" ", "_")
    filename = f"idb_export_{safe_name}.xlsx"

    return FileResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# ─── 静态文件 ────────────────────────────────

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """主页面"""
    from fastapi.responses import HTMLResponse
    html_file = static_dir / "index.html"
    if html_file.exists():
        content = html_file.read_text(encoding='utf-8')
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    return {"message": "idb API server running. Frontend not found."}


# ─── 启动入口 ────────────────────────────────

def main():
    import uvicorn
    init_config()
    print("🚀 idb Web 服务启动: http://localhost:8720")
    uvicorn.run(app, host="0.0.0.0", port=8720, log_level="info")


if __name__ == "__main__":
    main()
