"""向量嵌入器 — 为 Chunk 生成 Embedding"""

from idb.llm.client import LLMClient


def embed_chunks(
    chunks: list,
    client: LLMClient | None = None,
    batch_size: int = 20,
) -> list:
    """为 Chunk 列表生成嵌入向量。

    Args:
        chunks: Chunk 对象列表（需要有 .content 属性）或 dict 列表
        client: LLM 客户端（如果为 None 则自动创建）
        batch_size: 每批嵌入的文本数量

    Returns:
        输入 chunks 对应的向量列表，每个向量是 float 列表
    """
    if client is None:
        client = LLMClient()

    all_vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [_get_content(c) for c in batch]
        vectors = client.embed(texts)
        all_vectors.extend(vectors)

    return all_vectors


def _get_content(chunk) -> str:
    """兼容获取 chunk 的文本内容"""
    if hasattr(chunk, 'content'):
        return chunk.content
    elif isinstance(chunk, dict):
        return chunk.get('content', '')
    return str(chunk)
