"""提取器 — LLM 驱动的动态字段提取与结构化输出"""

import json
from typing import Optional

from idb.llm.client import LLMClient
from idb.config import get_config


def extract_and_structure(
    user_query: str,
    chunks: list[dict],
    client: LLMClient | None = None,
    target_fields: list[str] | None = None,
) -> dict:
    """从检索到的文档块中提取结构化数据。

    这是整个系统最核心的方法：它实现了"Schema-on-Read"。
    用户查询时不需要知道字段名，LLM 会自动：
    1. 推断需要提取哪些字段
    2. 从候选文档中逐字段提取
    3. 合并去重
    4. 输出结构化结果 + 来源标注 + 置信度

    Args:
        user_query: 用户的自然语言查询
        chunks: 检索到的候选文档块
        client: LLM 客户端
        target_fields: 可选预定义的字段列表（如果为None则自动推断）

    Returns:
        {
            "fields": [...],           # 提取的字段列表
            "records": [...],          # 结构化记录
            "source_files": [...],     # 来源文件列表
            "avg_confidence": float,   # 平均置信度
            "warnings": [...],         # 警告信息
        }
    """
    cfg = get_config()
    if client is None:
        client = LLMClient()

    # 0. 检测聚合查询（总数/总金额/合计/多少/共几份）
    agg_info = _detect_aggregate(user_query)

    # 1. 推断目标字段（如果未提供）
    if target_fields is None:
        target_fields = client.infer_target_fields(user_query)

    # 2. LLM 提取（0条时重试一次，缓解 LLM 随机性）
    raw_records = client.extract(
        user_query=user_query,
        target_fields=target_fields,
        chunks=chunks[:cfg.top_k_rerank],
    )
    if not raw_records:
        raw_records = client.extract(
            user_query=user_query,
            target_fields=target_fields,
            chunks=chunks[:cfg.top_k_rerank],
        )

    # 2.5 聚合查询：表格程序化聚合 > LLM聚合 > 通用程序化汇总
    if agg_info:
        agg_result = _compute_aggregate_table(user_query, chunks, agg_info)
        if agg_result:
            return agg_result
        agg_result = _compute_aggregate_llm(user_query, client, chunks, agg_info)
        if agg_result:
            return agg_result
        agg_result = _compute_aggregate(user_query, target_fields, raw_records, chunks, agg_info)
        if agg_result:
            return agg_result

    # 3. 后处理
    return _post_process(user_query, target_fields, raw_records, chunks)




def _compute_aggregate_table(
    query: str,
    chunks: list[dict],
    agg_info: dict,
) -> dict | None:
    """从结构化表格 chunk 中程序化提取金额列并汇总。

    表格 chunk 为管道分隔的 markdown 风格文本，列结构清晰，
    无需 LLM 即可精确聚合。这是聚合查询的首选方法。
    """
    import re

    # 只处理 table 类型 chunk
    table_chunks = [c for c in chunks if c.get("content_type") == "table"]
    if not table_chunks:
        return None

    for ch in table_chunks:
        content = ch.get("content", "")
        lines = content.strip().split("\n")
        if len(lines) < 3:
            continue

        # 找表头行（包含"合同"或"金额"或"序号"）
        header_idx = -1
        for i, line in enumerate(lines[:5]):
            if any(k in line for k in ['合同名称', '金额', '序号', '设备']):
                header_idx = i
                break
        if header_idx < 0:
            continue

        headers = [h.strip() for h in lines[header_idx].split('|')]
        # 找金额列，并识别列单位（万元）
        amount_col = -1
        amount_col_unit = ''  # '' | '万元'
        for ci, h in enumerate(headers):
            if any(k in h for k in ['中标', '金额', '成交', '总价']):
                amount_col = ci
                if '万元' in h or '万' in h:
                    amount_col_unit = '万元'
                break
        if amount_col < 0:
            continue

        # 重建跨行单元格：长合同名可能换行
        # 续行特征：不以数字序号开头，第一个'|'前是上一行合同名的续写
        data_lines = []
        for line in lines[header_idx+1:]:
            if not line.strip() or line.strip().startswith('---'):
                continue
            if data_lines and not line.strip()[:1].isdigit() and not line.strip().startswith('|'):
                # 续行：第一部分合并到上一行 cell[1]，其余列向后对齐
                parts = line.split('|', 1)
                if len(data_lines[-1]) > 1:
                    data_lines[-1][1] = data_lines[-1][1] + parts[0]
                if len(parts) > 1:
                    data_lines[-1].extend([c.strip() for c in parts[1].split('|')])
            else:
                data_lines.append([c.strip() for c in line.split('|')])

        values = []
        for cells in data_lines:
            if amount_col >= len(cells):
                continue
            # 只处理含数字序号（合同行）的数据行，跳过汇总行（如"设备总金额"）
            try:
                seq = int(cells[0].replace(',', ''))
            except (ValueError, TypeError):
                continue
            raw = cells[amount_col]
            if not raw:
                continue
            # 提取数值（处理 10.98万元 / 1,298,000.00 等）
            m = re.search(r'(\d[\d,]*\.?\d*)\s*(万元|万|元)?', raw)
            if not m:
                continue
            try:
                val = float(m.group(1).replace(',', ''))
                unit = m.group(2) or ''
                if '万' in unit or amount_col_unit == '万元':
                    val *= 10000
                # 转换为元后，合同金额通常 >= 1000 元
                if val >= 1000:
                    values.append(val)
            except (ValueError, TypeError):
                continue

        if not values:
            continue

        agg_type = agg_info["type"]
        if agg_type == 'count':
            label = "数量/记录数"
            result_text = str(len(values))
            value = len(values)
        elif agg_type == 'avg':
            label = "平均金额"
            value = sum(values) / len(values)
            result_text = f"{value:,.2f}"
        else:
            label = "总金额"
            value = sum(values)
            result_text = f"{value:,.2f}"

        return {
            "query": query,
            "fields": [label],
            "records": [{
                label: result_text,
                "来源": ch.get("file_name", ""),
                "_source": f"表格聚合 {len(values)} 条记录（{headers[amount_col]}）",
                "_confidence": 1.0,
            }],
            "source_files": [ch.get("file_name", "")],
            "avg_confidence": 1.0,
            "warnings": [],
            "aggregate": {"type": agg_type, "n": len(values), "value": value},
        }
    return None


def _compute_aggregate_llm(
    query: str,
    client: "LLMClient",
    chunks: list[dict],
    agg_info: dict,
) -> dict | None:
    """使用 LLM 提取金额值并汇总（优先方法）。

    只将最相关文件（通常包含金额表格的文件）的 chunks 传给 LLM，
    避免 30 个无关 chunks 干扰数值提取。
    """
    # 优先 table 类型 chunk；否则取排名最前文件的所有 chunks
    table_chunks = [c for c in chunks if c.get("content_type") == "table"]
    if table_chunks:
        focus_chunks = table_chunks[:3]
    else:
        # 找出现次数最多的文件（最相关文件）
        from collections import Counter
        fn_counter = Counter(c.get("file_name", "") for c in chunks)
        if fn_counter:
            top_file = fn_counter.most_common(1)[0][0]
            focus_chunks = [c for c in chunks if c.get("file_name") == top_file][:5]
        else:
            focus_chunks = chunks[:5]

    try:
        agg = client.extract_aggregate(query, focus_chunks)
    except Exception:
        return None

    if not agg or not agg.get("values"):
        return None

    values = [float(v) for v in agg.get("values", []) if isinstance(v, (int, float))]
    if not values:
        return None

    total = float(agg.get("total", sum(values)))
    n = len(values)
    agg_type = agg_info["type"]

    if agg_type == "count":
        label = "数量/记录数"
        result_text = str(n)
        value = n
    elif agg_type == "avg":
        label = "平均金额"
        value = total / n if n else 0
        result_text = f"{value:,.2f}"
    else:
        label = "总金额"
        value = total
        result_text = f"{value:,.2f}"

    source_files = sorted({ch.get("file_name", "") for ch in chunks if ch.get("file_name")})
    return {
        "query": query,
        "fields": [label],
        "records": [{
            label: result_text,
            "来源": "；".join(source_files[:5]),
            "_source": f"LLM聚合 {n} 条记录",
            "_confidence": 0.95,
        }],
        "source_files": source_files,
        "avg_confidence": 0.95,
        "warnings": [],
        "aggregate": {"type": agg_type, "n": n, "value": value},
    }


def _detect_aggregate(query: str) -> dict | None:
    """检测聚合查询意图，返回聚合类型和字段提示。"""
    q = query.lower()
    aggregate_patterns = [
        ('sum', ['总金额', '总计', '合计', '总共', '总额', '总和', 'sum', 'total']),
        ('count', ['共有', '总共几', '多少份', '多少个', '几份', 'count', 'how many']),
        ('avg', ['平均', '均值', 'average', 'avg']),
    ]
    for agg_type, kws in aggregate_patterns:
        if any(kw in query for kw in kws):
            return {"type": agg_type, "keywords": [k for k in kws if k in query]}
    return None


def _extract_numbers(text: str) -> list[float]:
    """从文本中提取金额/数字（只提取明确金额上下文的值）。

    只匹配带 元/万元/¥/人民币 单位的数值，避免把合同编号、
    日期、电话等无关数字误当金额。
    """
    import re
    nums = []
    # 1) 显式单位：1,298,000元 / 1298000万元 / ¥1298000 / 人民币1298000元
    for m in re.finditer(r'(?:¥|￥|人民币)?\s*(\d[\d,]*\.?\d*)\s*(万元|万圆|万|元|圆|CNY)', text):
        try:
            val = float(m.group(1).replace(',', ''))
            unit = m.group(2)
            if '万' in unit:
                val *= 10000
            # 金额应 >= 1000 元（过滤 0.3, 3, 日期等）
            if val >= 1000:
                nums.append(val)
        except (ValueError, TypeError):
            continue
    # 2) 大写中文金额（如 柒拾玖万陆仟伍佰捌拾元）——简单转换
    for m in re.finditer(r'([零壹贰叁肆伍陆柒捌玖拾佰仟万亿元整]+)', text):
        try:
            val = _chinese_num_to_value(m.group(1))
            if val >= 1000:
                nums.append(val)
        except Exception:
            continue
    return nums


def _chinese_num_to_value(s: str) -> float:
    """将中文大写金额转为数值（支持到万亿元）。"""
    digits = {'零': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4,
              '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9}
    units = {'拾': 10, '佰': 100, '仟': 1000}
    big = {'万': 10000, '亿': 100000000}
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in digits:
            number = digits[ch]
        elif ch in units:
            section += number * units[ch]
            number = 0
        elif ch in big:
            total += (section + number) * big[ch]
            section = 0
            number = 0
        elif ch in ('元', '整', '圆'):
            total += section + number
    total += section + number
    return float(total)


def _compute_aggregate(
    query: str,
    fields: list[str],
    records: list[dict],
    chunks: list[dict],
    agg_info: dict,
) -> dict | None:
    """程序化聚合：从提取记录中收集数值并计算 sum/count/avg。"""
    import json

    agg_type = agg_info["type"]

    # 收集所有数值字段
    all_numbers = []
    field_names = []
    for rec in records:
        for k, v in rec.items():
            if k.startswith('_') or v is None:
                continue
            if isinstance(v, (int, float)):
                all_numbers.append(float(v))
                if k not in field_names:
                    field_names.append(k)
            elif isinstance(v, str):
                nums = _extract_numbers(v)
                if nums:
                    all_numbers.extend(nums)
                    if k not in field_names:
                        field_names.append(k)

    # 如果 LLM 提取的数值不足，直接从 chunk 文本中提取
    if not all_numbers:
        for ch in chunks:
            all_numbers.extend(_extract_numbers(ch.get("content", "")))

    if not all_numbers:
        return None

    # 计算聚合
    if agg_type == 'sum':
        value = sum(all_numbers)
        label = f"总{field_names[0] if field_names else '金额'}"
        result_text = f"{value:,.2f}"
    elif agg_type == 'count':
        value = len(all_numbers)
        label = "数量/记录数"
        result_text = f"{value}"
    else:  # avg
        value = sum(all_numbers) / len(all_numbers)
        label = f"平均{field_names[0] if field_names else '金额'}"
        result_text = f"{value:,.2f}"

    source_files = sorted({ch.get("file_name", "") for ch in chunks if ch.get("file_name")})
    return {
        "query": query,
        "fields": [label] + field_names,
        "records": [{
            label: result_text,
            "来源": "；".join(source_files[:5]),
            "_source": "自动聚合 " + " + ".join(f"{n:,.0f}" for n in all_numbers[:10]) + ("..." if len(all_numbers) > 10 else ""),
            "_confidence": 0.95 if all_numbers else 0.0,
        }],
        "source_files": source_files,
        "avg_confidence": 0.95 if all_numbers else 0.0,
        "warnings": [] if all_numbers else ["未找到可聚合的数值。"],
        "aggregate": {"type": agg_type, "n": len(all_numbers), "value": value},
    }


def _post_process(
    query: str,
    fields: list[str],
    records: list[dict],
    chunks: list[dict],
) -> dict:
    """后处理：验证、去重、统计"""

    if not records:
        return {
            "query": query,
            "fields": fields,
            "records": [],
            "source_files": [],
            "avg_confidence": 0.0,
            "warnings": ["未找到匹配的数据。请尝试调整查询范围或关键词。"],
        }

    # 标准化记录
    cleaned = []
    confidences = []
    source_files = set()

    for rec in records:
        if not isinstance(rec, dict):
            continue

        # 提取置信度
        conf = rec.pop("置信度", rec.pop("confidence", None))
        if conf is not None:
            try:
                confidences.append(float(conf))
            except (ValueError, TypeError):
                confidences.append(0.7)
        else:
            confidences.append(0.7)

        # 提取来源引用
        source = rec.pop("来源引用", rec.pop("source", rec.pop("来源", None)))
        rec["_source"] = source
        rec["_confidence"] = conf

        # 收集来源文件
        if isinstance(source, str):
            for ch in chunks:
                if ch["file_name"] in source:
                    source_files.add(ch["file_name"])
        elif isinstance(source, list):
            for s in source:
                if isinstance(s, dict):
                    source_files.add(s.get("file", ""))
                elif isinstance(s, str):
                    for ch in chunks:
                        if ch["file_name"] in s:
                            source_files.add(ch["file_name"])

        cleaned.append(rec)

    # 如果置信度提取失败，从 chunks 推断来源文件
    if not source_files:
        source_files = {ch["file_name"] for ch in chunks[:5]}

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    # 生成警告
    warnings = []
    if avg_conf < 0.7:
        warnings.append("⚠️ 整体置信度较低，建议人工核实原文。")
    low_conf_records = [i for i, c in enumerate(confidences) if c < 0.6]
    if low_conf_records:
        warnings.append(f"⚠️ 第 {low_conf_records} 条记录置信度偏低，可能不准确。")

    # 查询字段覆盖度
    covered = set()
    for rec in cleaned:
        for k, v in rec.items():
            if not k.startswith("_") and v is not None:
                covered.add(k)
    missing_fields = [f for f in fields if f not in covered and f not in covered]
    if missing_fields:
        warnings.append(f"ℹ️ 以下字段未找到数据: {missing_fields}")

    return {
        "query": query,
        "fields": fields,
        "records": cleaned,
        "source_files": sorted(source_files),
        "avg_confidence": avg_conf,
        "warnings": warnings,
    }
