"""CLI 入口 — idb 命令行工具

用法:
    idb init                     初始化数据库
    idb ingest <path>            导入文件或目录
    idb ask "<query>"            自然语言查询
    idb stats                    查看数据库统计
    idb web                      启动 Web 应用
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint

from idb.config import init_config, get_config

console = Console()


# ═══════════════════════════════════════════════════════════════
# 命令: init
# ═══════════════════════════════════════════════════════════════

@click.group()
@click.option('--db', 'db_path', default=None, help='数据库路径 (默认: ~/idb_data)')
@click.pass_context
def main(ctx, db_path):
    """idb — 智能无Schema个人数据库

    把文件丢进去，用自然语言问出来。
    """
    init_config(db_path=db_path)
    ctx.ensure_object(dict)


@main.command()
def init():
    """初始化数据库"""
    cfg = get_config()
    db_path = Path(cfg.db_path)

    if (db_path / "vectors").exists():
        console.print(f"[yellow]数据库已存在于 {db_path}[/yellow]")
        return

    db_path.mkdir(parents=True, exist_ok=True)
    (db_path / "vectors").mkdir(exist_ok=True)

    # 创建 .env 模板
    env_file = db_path / ".env"
    if not env_file.exists():
        env_file.write_text("""# idb 配置文件
# 至少需要设置一个 LLM API Key

# OpenAI (用于 Embedding)
OPENAI_API_KEY=sk-xxx

# Anthropic (用于 LLM 提取)
ANTHROPIC_API_KEY=sk-ant-xxx

# 可选：自定义 Embedding 模型
# IDB_EMBEDDING_MODEL=text-embedding-3-small
# IDB_LLM_MODEL=claude-sonnet-4-6
""")

    # 触发 LanceDB 创建
    from idb.storage.lancedb_store import LanceDBStore
    LanceDBStore()

    console.print(Panel.fit(
        f"[bold green]✅ 数据库已创建[/bold green]\n"
        f"路径: {db_path}\n"
        f"请在 {env_file} 中设置 API Key\n\n"
        f"[dim]然后运行: idb ingest <文件或目录>[/dim]",
        title="idb init",
    ))


# ═══════════════════════════════════════════════════════════════
# 命令: ingest
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.argument('target', type=click.Path(exists=True))
@click.option('--recursive/--no-recursive', default=True, help='递归处理子目录')
@click.option('--force', is_flag=True, help='强制重新摄入已存在的文件')
@click.option('--dry-run', is_flag=True, help='仅显示将摄入的文件，不实际执行')
def ingest(target, recursive, force, dry_run):
    """导入文件或目录到数据库"""
    from idb.ingestion.parser import parse_file
    from idb.ingestion.chunker import chunk_document
    from idb.ingestion.embedder import embed_chunks
    from idb.storage.lancedb_store import LanceDBStore
    from idb.llm.client import LLMClient

    target_path = Path(target)

    # 收集文件
    files_to_ingest = []
    if target_path.is_file():
        files_to_ingest = [target_path]
    else:
        pattern = '**/*' if recursive else '*'
        for f in target_path.glob(pattern):
            if f.is_file() and not f.name.startswith('.'):
                files_to_ingest.append(f)

    if not files_to_ingest:
        console.print("[yellow]未找到可导入的文件。[/yellow]")
        return

    console.print(f"🔍 扫描到 [bold]{len(files_to_ingest)}[/bold] 个文件")

    if dry_run:
        for f in files_to_ingest:
            console.print(f"  📄 {f}")
        return

    store = LanceDBStore()
    from idb.storage.sqlite_store import SQLiteStore
    sqlite = SQLiteStore()
    client = LLMClient()

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("摄入文件...", total=len(files_to_ingest))

        for file_path in files_to_ingest:
            fname = file_path.name
            progress.update(task, description=f"📄 {fname}")

            try:
                # Step 1: 解析
                parsed = parse_file(str(file_path))

                # Step 2: 去重检查
                if not force and store.file_exists_by_hash(parsed.hash):
                    skip_count += 1
                    progress.advance(task)
                    continue

                # Step 3: 分块
                file_id = str(uuid.uuid4())
                chunks = chunk_document(
                    file_id=file_id,
                    file_name=parsed.file_name,
                    text=parsed.text,
                    tables=parsed.tables,
                )

                if not chunks:
                    skip_count += 1
                    progress.advance(task)
                    continue

                # Step 4: 向量嵌入
                vectors = embed_chunks(chunks, client=client)

                # Step 5: 双路写入 (LanceDB + SQLite)
                store.add_file_record(
                    file_id=file_id,
                    path=parsed.file_path,
                    name=parsed.file_name,
                    file_hash=parsed.hash,
                    mime_type=parsed.mime_type,
                    size_bytes=parsed.size_bytes,
                    chunk_count=len(chunks),
                )
                store.add_chunks(chunks, vectors)

                sqlite.add_file(
                    file_id, parsed.file_path, parsed.file_name, parsed.hash,
                    parsed.mime_type, parsed.size_bytes, len(chunks),
                )
                sqlite.add_chunks(chunks)
                sqlite.log_ingest(parsed.file_name, "ingest", "success", f"{len(chunks)} chunks")

                success_count += 1
                total_chunks += len(chunks)

            except Exception as e:
                fail_count += 1
                console.print(f"[red]  ❌ {fname}: {e}[/red]")

            progress.advance(task)

    # 摘要
    console.print()
    console.print(Panel.fit(
        f"[bold green]✅ 摄入完成[/bold green]\n\n"
        f"成功: {success_count} 个文件, {total_chunks} 个语义块\n"
        f"跳过: {skip_count} 个文件 (已存在)\n"
        + (f"失败: {fail_count} 个文件\n" if fail_count else ""),
        title="摄入报告",
    ))


# ═══════════════════════════════════════════════════════════════
# 命令: ask
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.argument('query', type=str)
@click.option('--top', 'top_k', default=None, type=int, help='检索候选数量')
@click.option('--json', 'output_json', is_flag=True, help='以 JSON 格式输出')
@click.option('--fields', '-f', multiple=True, help='指定提取的字段 (可多次使用)')
def ask(query, top_k, output_json, fields):
    """用自然语言查询数据库"""
    from idb.query.retriever import retrieve
    from idb.query.extractor import extract_and_structure
    from idb.storage.lancedb_store import LanceDBStore
    from idb.llm.client import LLMClient

    import time
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: 分析意图
        task = progress.add_task("[cyan]① 分析查询意图...", total=None)

        store = LanceDBStore()
        from idb.storage.sqlite_store import SQLiteStore
        sqlite = SQLiteStore()
        client = LLMClient()

        # Step 2: 多路检索
        progress.update(task, description="[cyan]② 多路语义检索 (向量+FTS+元数据)...")
        chunks = retrieve(
            query=query,
            store=store,
            sqlite=sqlite,
            client=client,
            top_k=top_k,
        )

        if not chunks:
            progress.update(task, description="[yellow]❌ 未找到相关文档")
            console.print("[yellow]未找到相关文档。请尝试调整查询范围或关键词。[/yellow]")
            return

        t1 = time.time()
        progress.update(task, description=f"[cyan]③ 检索完成: {len(chunks)} 个候选 ({t1-t0:.1f}s) → LLM提取中...")

    # Step 3: LLM 提取
    target_fields = list(fields) if fields else None
    result = extract_and_structure(
        user_query=query,
        chunks=chunks,
        client=client,
        target_fields=target_fields,
    )

    # Phase 3: 输出
    if output_json:
        import json
        console.print_json(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_table(result)


def _print_table(result: dict):
    """以 Rich 表格形式输出结果"""
    records = result.get("records", [])
    fields = result.get("fields", [])
    source_files = result.get("source_files", [])
    avg_conf = result.get("avg_confidence", 0.0)
    warnings = result.get("warnings", [])

    if not records:
        console.print("[yellow]未提取到结构化数据。[/yellow]")
        for w in warnings:
            console.print(f"  {w}")
        return

    # 确定表格列：所有非 _ 前缀的字段
    display_fields = []
    for rec in records:
        for k in rec:
            if not k.startswith("_") and k not in display_fields:
                display_fields.append(k)

    if not display_fields:
        display_fields = ["数据"]

    # 构建表格
    table = Table(title=f"查询结果: {result['query'][:40]}...")
    table.add_column("#", style="dim", width=3)

    for f in display_fields:
        table.add_column(f, style="cyan" if f in fields else "")
    table.add_column("置信度", style="yellow", width=8)

    for i, rec in enumerate(records):
        row = [str(i + 1)]
        for f in display_fields:
            val = rec.get(f, "")
            if val is None:
                row.append("[dim]N/A[/dim]")
            else:
                row.append(str(val)[:60])
        conf = rec.get("_confidence", 0.0)
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
        if isinstance(conf, (int, float)) and conf < 0.6:
            conf_str = f"[red]{conf_str}[/red]"
        elif isinstance(conf, (int, float)) and conf < 0.8:
            conf_str = f"[yellow]{conf_str}[/yellow]"
        else:
            conf_str = f"[green]{conf_str}[/green]"
        row.append(conf_str)
        table.add_row(*row)

    console.print(table)

    # 来源 & 警告
    console.print()
    if source_files:
        console.print(f"📂 来源文件: [dim]{', '.join(source_files[:5])}[/dim]")
    console.print(f"📊 平均置信度: [bold]{avg_conf:.0%}[/bold] | 共 {len(records)} 条记录")

    if warnings:
        console.print()
        for w in warnings:
            console.print(f"  {w}")


# ═══════════════════════════════════════════════════════════════
# 命令: web
# ═══════════════════════════════════════════════════════════════

@main.command()
@click.option('--port', '-p', default=8720, help='服务端口 (默认 8720)')
@click.option('--host', '-h', default='127.0.0.1', help='绑定地址')
def web(port, host):
    """启动 Web 管理界面"""
    from idb.web.server import app
    import uvicorn

    console.print(Panel.fit(
        f"[bold green]🚀 idb Web 服务[/bold green]\n\n"
        f"本地访问: [bold]http://{host}:{port}[/bold]\n"
        f"按 Ctrl+C 停止服务",
        title="idb web"
    ))
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ═══════════════════════════════════════════════════════════════
# 命令: stats
# ═══════════════════════════════════════════════════════════════

@main.command()
def stats():
    """查看数据库统计信息"""
    from idb.storage.lancedb_store import LanceDBStore

    store = LanceDBStore()
    s = store.get_file_stats()

    table = Table(title="数据库统计")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("文件数量", str(s["file_count"]))
    table.add_row("语义块数量", str(s["chunk_count"]))
    table.add_row("总大小", f"{s['total_size_mb']} MB")
    table.add_row("数据库路径", str(Path(get_config().db_path).resolve()))

    console.print(table)


if __name__ == "__main__":
    main()
