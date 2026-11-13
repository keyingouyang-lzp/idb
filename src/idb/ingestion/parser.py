"""文件解析器 — 将多种格式文件提取为纯文本"""

import hashlib
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedDocument:
    """解析后的文档"""
    file_path: str
    file_name: str
    mime_type: str
    text: str
    hash: str                          # SHA256
    size_bytes: int
    tables: list[dict] = field(default_factory=list)   # 提取的表格数据
    metadata: dict = field(default_factory=dict)        # 文件级元数据


def parse_file(file_path: str) -> ParsedDocument:
    """解析单个文件，自动识别格式并提取文本。

    支持的格式:
    - .txt, .md, .markdown → 直接读取
    - .pdf → PyMuPDF
    - .csv → 直接读取为文本+表格

    Args:
        file_path: 文件路径

    Returns:
        ParsedDocument: 解析结果，包含纯文本和元数据
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    file_name = path.name
    size_bytes = path.stat().st_size

    # 计算文件哈希
    file_hash = _compute_hash(file_path)

    # 按扩展名分派解析器
    if ext in ('.txt', '.md', '.markdown', '.csv', '.log', '.json', '.xml',
               '.yml', '.yaml', '.toml', '.ini', '.cfg', '.py', '.js',
               '.ts', '.html', '.css', '.sh', '.bash', '.zsh', '.rst'):
        text, tables, meta = _parse_text_file(file_path, ext)
        mime_type = f"text/{ext[1:]}" if ext != '.md' else 'text/markdown'

    elif ext == '.pdf':
        text, tables, meta = _parse_pdf(file_path)
        mime_type = 'application/pdf'

    elif ext in ('.docx', '.doc'):
        text, tables, meta = _parse_docx(file_path, ext)
        mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

    elif ext in ('.xlsx', '.xls'):
        text, tables, meta = _parse_xlsx(file_path, ext)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    elif ext == '.pptx':
        text, tables, meta = _parse_pptx(file_path)
        mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'

    else:
        # 尝试作为文本文件读取
        try:
            text, tables, meta = _parse_text_file(file_path, ext)
            mime_type = f'text/{ext[1:]}" if ext else "text/plain'
        except UnicodeDecodeError:
            raise ValueError(f"不支持的文件格式: {ext} (或以二进制形式读取失败)")

    return ParsedDocument(
        file_path=str(path.resolve()),
        file_name=file_name,
        mime_type=mime_type,
        text=text,
        hash=file_hash,
        size_bytes=size_bytes,
        tables=tables,
        metadata=meta,
    )


def _compute_hash(file_path: str) -> str:
    """计算文件 SHA256"""
    sha = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def _parse_text_file(file_path: str, ext: str) -> tuple[str, list[dict], dict]:
    """解析纯文本文件"""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    tables = []
    if ext == '.csv':
        import csv, io
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        rows = [row for row in reader]
        if headers:
            tables.append({"headers": headers, "rows": rows, "source": "csv"})

    meta = {
        "line_count": text.count('\n') + 1,
        "char_count": len(text),
    }
    return text, tables, meta


def _parse_pdf(file_path: str) -> tuple[str, list[dict], dict]:
    """解析 PDF 文件，自动检测扫描件并启用 OCR"""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    pages_text = []
    tables = []
    meta = {"page_count": len(doc), "title": doc.metadata.get("title", ""), "ocr_used": False}

    for page_idx, page in enumerate(doc):
        # 提取文本
        page_text = page.get_text("text")
        pages_text.append(page_text)

        # 尝试提取表格
        try:
            tabs = page.find_tables()
            for tab_idx, tab in enumerate(tabs):
                rows = tab.extract()
                if rows:
                    tables.append({
                        "page": page_idx + 1,
                        "table_index": tab_idx,
                        "headers": rows[0] if rows else [],
                        "rows": rows[1:] if len(rows) > 1 else [],
                    })
        except Exception:
            pass

    doc.close()
    full_text = "\n\n".join(pages_text)

    # 检测是否为扫描件（文本极少 = 很可能是图片型 PDF）
    avg_chars_per_page = len(full_text.strip()) / max(len(pages_text), 1)
    if avg_chars_per_page < 100:
        ocr_text = _ocr_pdf(file_path, len(pages_text))
        if ocr_text and len(ocr_text) > len(full_text) * 2:
            full_text = ocr_text
            meta["ocr_used"] = True

    return full_text, tables, meta


def _ocr_pdf(file_path: str, page_count: int) -> str:
    """对扫描版 PDF 执行 OCR 文本提取"""
    import os
    os.environ.setdefault('TESSDATA_PREFIX', '/opt/homebrew/share/tessdata/')
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    try:
        from pdf2image import convert_from_path
    except ImportError:
        return ""

    try:
        import pytesseract
    except ImportError:
        return ""

    text_parts = []
    # OCR 前 5 页作为采样；若质量好则继续
    sample_pages = min(page_count, 5)

    try:
        images = convert_from_path(
            file_path,
            first_page=1,
            last_page=sample_pages,
            dpi=180,
        )
    except Exception:
        return ""

    quality_score = 0
    for i, img in enumerate(images):
        try:
            img.thumbnail((2600, 2600))
            page_text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
            text_parts.append(page_text)
            # 简单质量评估：有效中文字符比例
            chinese_chars = sum(1 for c in page_text if '一' <= c <= '鿿')
            quality_score += 1 if chinese_chars > 20 else 0
        except Exception:
            text_parts.append("")

    # 如果前 5 页质量太差（< 2 页有中文），放弃 OCR
    if quality_score < 2 and page_count > 5:
        return "\n\n".join(text_parts) + "\n\n[OCR 质量较低，仅提取了前5页]"

    # 质量还行，继续 OCR 剩余页面
    if page_count > sample_pages:
        try:
            rest_images = convert_from_path(
                file_path,
                first_page=sample_pages + 1,
                last_page=page_count,
                dpi=180,
            )
            for img in rest_images:
                try:
                    img.thumbnail((2600, 2600))
                    page_text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6')
                    text_parts.append(page_text)
                except Exception:
                    text_parts.append("")
        except Exception:
            pass

    return "\n\n".join(text_parts)


def _parse_docx(file_path: str, ext: str) -> tuple[str, list[dict], dict]:
    """解析 Word 文档 (.docx / .doc)"""
    tables = []
    meta = {}

    if ext == '.docx':
        from docx import Document
        doc = Document(file_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        for tab_idx, table in enumerate(doc.tables):
            rows_data = []
            for row in table.rows:
                cells = [cell.text for cell in row.cells]
                rows_data.append(cells)
            if rows_data:
                tables.append({
                    "table_index": tab_idx,
                    "headers": rows_data[0] if rows_data else [],
                    "rows": rows_data[1:] if len(rows_data) > 1 else [],
                    "source": "docx",
                })

        text = "\n\n".join(paragraphs)
        for tab in tables:
            text += "\n\n[表格]\n"
            if tab["headers"]:
                text += " | ".join(str(h) for h in tab["headers"]) + "\n"
            for row in tab["rows"]:
                text += " | ".join(str(c) for c in row) + "\n"

        meta["paragraph_count"] = len(paragraphs)
        meta["table_count"] = len(doc.tables)

    elif ext == '.doc':
        import subprocess
        try:
            result = subprocess.run(
                ['textutil', '-convert', 'txt', '-stdout', file_path],
                capture_output=True, text=True, timeout=30
            )
            text = result.stdout
            meta["converter"] = "textutil"
        except Exception:
            try:
                result = subprocess.run(
                    ['antiword', file_path],
                    capture_output=True, text=True, timeout=30
                )
                text = result.stdout
                meta["converter"] = "antiword"
            except Exception:
                raise ValueError("无法解析 .doc 文件: 请安装 antiword (brew install antiword)")

    return text, tables, meta


def _parse_xlsx(file_path: str, ext: str) -> tuple[str, list[dict], dict]:
    """解析 Excel 表格 (.xlsx / .xls)"""
    tables = []
    text_parts = []
    meta = {}

    if ext == '.xlsx':
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        meta["sheet_names"] = wb.sheetnames

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows_data = []
            for row in ws.iter_rows(values_only=True):
                rows_data.append([str(c) if c is not None else "" for c in row])
            if rows_data:
                text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")
                for row in rows_data:
                    text_parts.append(" | ".join(row))
                tables.append({
                    "sheet": sheet_name,
                    "headers": rows_data[0] if rows_data else [],
                    "rows": rows_data[1:] if len(rows_data) > 1 else [],
                    "source": "xlsx",
                })
        wb.close()

    elif ext == '.xls':
        import xlrd
        wb = xlrd.open_workbook(file_path)
        meta["sheet_names"] = wb.sheet_names()
        for sheet_name in wb.sheet_names():
            ws = wb.sheet_by_name(sheet_name)
            rows_data = []
            for row_idx in range(ws.nrows):
                rows_data.append([str(ws.cell_value(row_idx, col_idx)) for col_idx in range(ws.ncols)])
            if rows_data:
                text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")
                for row in rows_data:
                    text_parts.append(" | ".join(row))
                tables.append({
                    "sheet": sheet_name,
                    "headers": rows_data[0] if rows_data else [],
                    "rows": rows_data[1:] if len(rows_data) > 1 else [],
                    "source": "xls",
                })

    text = "\n".join(text_parts)
    meta["sheet_count"] = len(tables)
    return text, tables, meta


def _parse_pptx(file_path: str) -> tuple[str, list[dict], dict]:
    """解析 PowerPoint 演示文稿 (.pptx)"""
    from pptx import Presentation
    prs = Presentation(file_path)
    slides_text = []
    tables = []
    meta = {"slide_count": len(prs.slides)}

    for slide_idx, slide in enumerate(prs.slides):
        slide_parts = [f"\n--- Slide {slide_idx + 1} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        slide_parts.append(para.text)
            if shape.has_table:
                table = shape.table
                rows_data = []
                for row in table.rows:
                    cells = [cell.text for cell in row.cells]
                    rows_data.append(cells)
                if rows_data:
                    slide_parts.append("[表格] " + (" | ".join(rows_data[0]) if rows_data else ""))
                    for row in rows_data[1:]:
                        slide_parts.append(" | ".join(row))
                    tables.append({
                        "slide": slide_idx + 1,
                        "headers": rows_data[0] if rows_data else [],
                        "rows": rows_data[1:] if len(rows_data) > 1 else [],
                        "source": "pptx",
                    })
        slides_text.append("\n".join(slide_parts))

    text = "\n\n".join(slides_text)
    meta["table_count"] = len(tables)
    return text, tables, meta
