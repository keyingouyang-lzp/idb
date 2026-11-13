"""智能分块器 — 将文档文本切分为适合检索的语义块"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from idb.config import get_config


@dataclass
class Chunk:
    """一个文档块"""
    id: str
    file_id: str              # 关联的文件ID（在存储层分配）
    file_name: str
    chunk_index: int
    content: str
    content_type: str = "text"     # text | table | code
    token_count: int = 0
    location: str = ""             # 如 "paragraphs 5-8"
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_type": self.content_type,
            "token_count": self.token_count,
            "location": self.location,
        }


def chunk_document(
    file_id: str,
    file_name: str,
    text: str,
    tables: list[dict] | None = None,
) -> list[Chunk]:
    """将解析后的文档文本切分为语义块。

    策略：
    1. 表格内容单独成块（保持完整性）
    2. 文本按段落边界切分
    3. 合并短段落直到达到目标大小
    4. 拆分过长段落
    5. 块间保持 20% 重叠

    Args:
        file_id: 文件ID
        file_name: 文件名
        text: 文档纯文本
        tables: 解析出的表格列表（可选）

    Returns:
        排序后的 Chunk 列表
    """
    cfg = get_config()
    chunk_size = cfg.chunk_size
    overlap = cfg.chunk_overlap

    chunks: list[Chunk] = []

    # ── 1. 表格块 ──
    if tables:
        for tab in tables:
            tab_text = _format_table(tab)
            if tab_text.strip():
                chunks.append(_make_chunk(
                    file_id, file_name, len(chunks),
                    tab_text, content_type="table",
                    location=f"table at page {tab.get('page', '?')}"
                ))

    # ── 2. 文本块 ──
    # 按段落分割
    paragraphs = _split_paragraphs(text)

    # 按自然语义边界合并/拆分
    text_chunks = _merge_and_split_paragraphs(
        paragraphs, chunk_size, overlap
    )

    for tc in text_chunks:
        chunks.append(_make_chunk(
            file_id, file_name, len(chunks),
            tc["text"], content_type="text",
            location=tc.get("location", "")
        ))

    # ── 3. 链接前后块 ──
    for i in range(len(chunks)):
        if i > 0:
            chunks[i].prev_chunk_id = chunks[i-1].id
        if i < len(chunks) - 1:
            chunks[i].next_chunk_id = chunks[i+1].id

    return chunks


def _make_chunk(
    file_id: str,
    file_name: str,
    chunk_index: int,
    content: str,
    content_type: str = "text",
    location: str = "",
) -> Chunk:
    """创建 Chunk 实例"""
    return Chunk(
        id=str(uuid.uuid4()),
        file_id=file_id,
        file_name=file_name,
        chunk_index=chunk_index,
        content=content.strip(),
        content_type=content_type,
        token_count=_estimate_tokens(content),
        location=location,
    )


def _split_paragraphs(text: str) -> list[str]:
    """按段落分割文本"""
    # 先按空行分割
    raw_paras = re.split(r'\n\s*\n', text)
    # 过滤纯空白段落
    return [p.strip() for p in raw_paras if p.strip()]


def _merge_and_split_paragraphs(
    paragraphs: list[str],
    target_size: int,
    overlap: int,
) -> list[dict]:
    """合并短段落、拆分长段落。

    使用近似字符数作为大小度量（1 char ≈ 0.5 token for 中文,
    1 char ≈ 0.25 token for 英文，取折中 ≈ 0.4 token/char）。
    """
    char_target = int(target_size / 0.4)   # 目标字符数
    char_overlap = int(overlap / 0.4)       # 重叠字符数
    char_min = char_target // 2             # 最小块大小

    result = []
    buffer = ""
    buffer_start = 0
    para_idx = 0

    def flush():
        nonlocal buffer, buffer_start
        if buffer.strip():
            result.append({
                "text": buffer.strip(),
                "location": f"paragraphs {buffer_start+1}-{para_idx}",
            })
            buffer = ""
            buffer_start = para_idx

    while para_idx < len(paragraphs):
        para = paragraphs[para_idx]

        # 单个段落超长 → 拆分
        if len(para) > char_target * 1.5:
            flush()
            result.extend(_split_long_paragraph(para, char_target, char_overlap, para_idx))
            buffer_start = para_idx + 1
            para_idx += 1
            continue

        # 加入 buffer
        candidate = (buffer + "\n\n" + para).strip() if buffer else para

        if len(candidate) > char_target and len(buffer) >= char_min:
            # buffer 已经够大，flush 然后重新开始
            flush()
            buffer = para
            buffer_start = para_idx
        else:
            buffer = candidate

        para_idx += 1

    flush()
    return result


def _split_long_paragraph(
    para: str,
    char_target: int,
    char_overlap: int,
    para_idx: int,
) -> list[dict]:
    """将超长段落拆分为多个重叠的块"""
    results = []
    start = 0
    part = 0
    while start < len(para):
        end = min(start + char_target, len(para))
        # 尝试在句号处截断
        if end < len(para):
            for sep in ['。', '；', '. ', '; ', '\n']:
                last_sep = para.rfind(sep, start, end)
                if last_sep > start + char_target // 2:
                    end = last_sep + 1
                    break

        results.append({
            "text": para[start:end].strip(),
            "location": f"paragraph {para_idx+1}, part {part+1}",
        })

        if end >= len(para):
            break
        start = end - char_overlap
        part += 1

    return results


def _format_table(tab: dict) -> str:
    """将表格 dict 格式化为可读文本"""
    headers = tab.get("headers", [])
    rows = tab.get("rows", [])

    lines = []
    if headers:
        lines.append(" | ".join(str(h) for h in headers))
        lines.append(" | ".join("---" for _ in headers))
    for row in rows:
        lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """估算 token 数（中英文混合）"""
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    other_chars = len(text) - chinese_chars
    # 中文字符 ≈ 1.5 token, 其他 ≈ 0.3 token
    return int(chinese_chars * 1.5 + other_chars * 0.3)
