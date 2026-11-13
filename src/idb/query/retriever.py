"""检索器 — 多路召回 + RRF 融合"""

from idb.llm.client import LLMClient
from idb.storage.lancedb_store import LanceDBStore
from idb.storage.sqlite_store import SQLiteStore
from idb.config import get_config


def retrieve(
    query: str,
    store: LanceDBStore | None = None,
    sqlite: SQLiteStore | None = None,
    client: LLMClient | None = None,
    top_k: int | None = None,
    use_cache: bool = True,
    filters: dict | None = None,
) -> list[dict]:
    """多路召回 + RRF 融合。

    多路搜索：
    1. 向量语义搜索 (LanceDB)
    2. 全文关键词搜索 (SQLite FTS5)
    3. 元数据精确匹配 (SQLite, 仅在调用方给出显式过滤条件时才参与)

    然后用 RRF (Reciprocal Rank Fusion) 合并排序。

    Args:
        query: 用户自然语言查询
        store: LanceDB 存储
        sqlite: SQLite 存储
        client: LLM 客户端
        top_k: 最终返回数量
        use_cache: 是否使用缓存
        filters: 显式元数据过滤条件（date_from/date_to/mime_type/content_type）。
                 为空时不参与融合 —— 空过滤会退化成"按 rowid 取前 N 个 chunk"，
                 其结果与查询无关，相当于给 RRF 叠加一个恒定先验。

    Returns:
        排序后的相关 chunk 列表
    """
    cfg = get_config()
    if store is None:
        store = LanceDBStore()
    if sqlite is None:
        sqlite = SQLiteStore()
    if client is None:
        client = LLMClient()
    if top_k is None:
        top_k = cfg.top_k_retrieve

    # 检查缓存
    if use_cache:
        cached = sqlite.cache_get(query)
        if cached:
            return cached.get("chunks", [])

    # 1. 向量搜索（扩大召回池，保证召回充足）
    query_vec = client.embed_single(query)
    vec_results = store.search(query_vec, top_k=top_k * 3)

    # 2. 全文搜索（trigram 中文分词 + LIKE 补充）
    fts_results = sqlite.search_fts(query, limit=top_k * 2)
    for r in fts_results:
        r["relevance"] = 0.5  # 基础分

    # 3. 元数据搜索：仅在调用方给出显式过滤条件时才参与融合。
    #    历史实现传入空 dict，sqlite.search_metadata 会退化为
    #    "WHERE 1=1 LIMIT N"，即对任意查询都返回同一批 chunk；
    #    这在真实语料上把 MRR@5 从 0.857 拉低到 0.600 且不增加召回。
    meta_results = sqlite.search_metadata(filters, limit=top_k) if filters else []
    for r in meta_results:
        r["relevance"] = 0.3

    # RRF 融合
    merged = _rrf_fuse(vec_results, fts_results, meta_results, k=60)

    # 动态多样化：根据语料文件数自动计算每文件上限，确保覆盖
    all_files = set()
    for r in merged:
        all_files.add(r.get("file_name", "unknown"))
    n_files = max(1, len(all_files))
    max_per_file = max(3, min(10, top_k // n_files + 2))
    merged = _diversify(merged, max_per_file=max_per_file, top_k=top_k)

    # 覆盖保证：确保不同文件的 top chunk 都进入候选
    merged = _ensure_coverage(merged, vec_results + fts_results, top_k=top_k)

    # 财务数据增强：含金额/数字的 chunk 提权。
    # 默认关闭 —— 该权重（单项最高 +0.008）与 RRF 相邻名次间距（~0.0003）
    # 量级不匹配，实测会把相关 chunk 挤出 top-5（MRR@5 0.857 -> 0.762）
    # 而对端到端抽取成功率没有贡献；需要时可设 IDB_FINANCIAL_BOOST=1 打开。
    if cfg.financial_boost:
        merged = _boost_financial(merged, query)

    # 将 cosine distance 转为 relevance
    for r in merged:
        score = r.get("score", 1.0)
        r["relevance"] = round(1.0 - min(score, 2.0) / 2.0, 4)

    # 写缓存
    if use_cache:
        sqlite.cache_set(query, {"chunks": merged})

    return merged


def _rrf_fuse(*result_lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion 多路结果融合。

    RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: 各路搜索结果列表
        k: 平滑常数（默认 60）

    Returns:
        按 RRF 分数降序排列的结果列表
    """
    scores = {}
    items = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            chunk_id = item.get("id", item.get("chunk_id", str(rank)))
            rrf = 1.0 / (k + rank + 1)
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                items[chunk_id] = item
            scores[chunk_id] += rrf
            # 保留最高分的那次结果
            if item.get("score", item.get("relevance", 0)) > items[chunk_id].get("score", items[chunk_id].get("relevance", 0)):
                items[chunk_id] = item

    # 按 RRF 分数排序
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    result = []
    for cid in sorted_ids:
        item = items[cid]
        item["rrf_score"] = round(scores[cid], 4)
        result.append(item)

    return result


def _boost_financial(results: list[dict], query: str) -> list[dict]:
    """对金额相关查询，提升包含数字/金额模式的 chunk 排名。

    OCR 文本中的金额 chunk 往往语义向量相似度偏低（因为噪声多），
    但对金额查询至关重要，需要轻度提权。
    """
    amount_keywords = ['金额', '总价', '价格', '万', '元', '¥', '款', '报价', '中标价']
    is_amount_query = any(kw in query for kw in amount_keywords)

    if not is_amount_query:
        return results

    import re
    amount_pattern = re.compile(r'\d[\d,.]*\s*(?:元|万|千|百|USD|CNY|¥|$)')

    for item in results:
        content = item.get("content", "")
        boost = 0.0
        # 包含金额关键词
        for kw in amount_keywords:
            if kw in content:
                boost += 0.003
        # 包含明显的金额数字模式
        if amount_pattern.search(content):
            boost += 0.005
        if boost > 0:
            item["rrf_score"] = item.get("rrf_score", 0) + boost
            # 也加一个标记
            item["relevance"] = min(1.0, item.get("relevance", 0) + boost)

    # 按更新后的 rrf_score 重排
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return results


def _ensure_coverage(results: list[dict], candidates: list[dict], top_k: int = 20) -> list[dict]:
    """覆盖保证：确保出现在候选池中的文件都有至少 1 个 chunk 进入最终结果。

    当 top_k 窗口较小时，高排名的单文件可能挤出其他文件的 chunks。
    该函数优先为候选池中的每个文件保留 1 个最高分 chunk，再填充其余名额。
    """
    if len(results) <= 2:
        return results

    # 收集候选池中的文件及其最佳 chunk
    file_best: dict[str, dict] = {}
    for r in candidates:
        fn = r.get("file_name", "unknown")
        if fn not in file_best or r.get("rrf_score", r.get("score", 0)) > file_best[fn].get("rrf_score", file_best[fn].get("score", 0)):
            file_best[fn] = r

    # 当前结果中已覆盖的文件
    covered = set(r.get("file_name", "unknown") for r in results)

    # 为未覆盖文件插入其最佳 chunk
    result = list(results)
    for fn, best in file_best.items():
        if fn not in covered and len(result) < top_k:
            result.append(best)
            covered.add(fn)

    # 去重（按 chunk id）
    seen = set()
    deduped = []
    for r in result:
        rid = r.get("id", r.get("chunk_id"))
        if rid not in seen:
            seen.add(rid)
            deduped.append(r)
    return deduped[:top_k]


def _diversify(results: list[dict], max_per_file: int = 8, top_k: int = 20) -> list[dict]:
    """确保不同文件的 chunks 都有机会进入 LLM 上下文。

    策略：保持排名靠前的 chunks，但限制每个文件的数量，
    为其他文件的 top chunks 腾出空间。
    """
    file_counts: dict[str, int] = {}
    diversified = []
    overflow = []

    for r in results:
        fn = r.get("file_name", "unknown")
        count = file_counts.get(fn, 0)
        if count < max_per_file:
            diversified.append(r)
            file_counts[fn] = count + 1
        else:
            overflow.append(r)

    # 如果有多余名额，从溢出的里面补
    while len(diversified) < top_k and overflow:
        diversified.append(overflow.pop(0))

    return diversified[:top_k]


def search_hybrid(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """简化接口：直接多路检索"""
    return retrieve(query, top_k=top_k)
