"""LLM 统一客户端 — 封装 Embedding 和 Chat 调用"""

import json
import os
from typing import Optional

from idb.config import get_config


def _get_text_from_response(message) -> str:
    """从 Anthropic 响应中提取文本，兼容 ThinkingBlock 和 TextBlock"""
    parts = []
    for block in message.content:
        if hasattr(block, 'text'):
            parts.append(block.text)
        elif hasattr(block, 'thinking'):
            # DeepSeek thinking block — 跳过思考内容
            pass
        else:
            # 尝试转为字符串
            parts.append(str(block))
    return "\n".join(parts)


class LLMClient:
    """统一的 LLM 调用接口。

    职责：
    - embed(): 文本 → 向量 (优先 API，回退本地模型)
    - extract(): 从检索结果中提取结构化数据 (Anthropic)
    """

    def __init__(self):
        self.cfg = get_config()
        self.cfg.ensure_api_keys()
        self._openai_client = None
        self._anthropic_client = None
        self._local_embed_model = None

    # ─── Embedding ───────────────────────────────────────

    @property
    def openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            kwargs = {}
            if self.cfg.openai_api_key:
                kwargs["api_key"] = self.cfg.openai_api_key
            if self.cfg.openai_base_url:
                kwargs["base_url"] = self.cfg.openai_base_url
            self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def _get_local_model(self):
        """懒加载本地 sentence-transformers 模型"""
        if self._local_embed_model is None:
            # 使用 HF 镜像 (国内网络加速)
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("IDB_LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
            self._local_embed_model = SentenceTransformer(model_name)
            # 兼容新旧版本 sentence-transformers API
            dim_method = getattr(self._local_embed_model, 'get_embedding_dimension',
                                 getattr(self._local_embed_model, 'get_sentence_embedding_dimension', None))
            if dim_method:
                dim = dim_method()
                self.cfg.embedding_dim = int(dim) if not isinstance(dim, int) else dim
        return self._local_embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转为向量列表。

        优先使用 OpenAI API，失败或无 API key 时回退到本地模型。

        Args:
            texts: 待嵌入的文本列表

        Returns:
            等长的向量列表，每个向量是 float 列表
        """
        if not texts:
            return []

        # 尝试 API
        if self.cfg.openai_api_key:
            try:
                resp = self.openai.embeddings.create(
                    model=self.cfg.embedding_model,
                    input=texts,
                )
                dim = len(resp.data[0].embedding)
                self.cfg.embedding_dim = dim
                return [d.embedding for d in resp.data]
            except Exception:
                pass  # 回退到本地模型

        # 本地模型
        model = self._get_local_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """嵌入单个文本"""
        return self.embed([text])[0]

    # ─── Chat / Extract ─────────────────────────────────

    @property
    def anthropic(self):
        if self._anthropic_client is None:
            from anthropic import Anthropic
            kwargs = {"api_key": self.cfg.anthropic_api_key}
            if self.cfg.anthropic_base_url:
                kwargs["base_url"] = self.cfg.anthropic_base_url
            self._anthropic_client = Anthropic(**kwargs)
        return self._anthropic_client

    def extract(
        self,
        user_query: str,
        target_fields: list[str],
        chunks: list[dict],
        *,
        model: Optional[str] = None,
    ) -> list[dict]:
        """从检索到的 chunks 中提取结构化数据。

        这是查询流水线最核心的一步：将非结构化文档内容
        转化为符合用户查询需求的结构化表格数据。

        Args:
            user_query: 用户原始查询
            target_fields: 需要提取的字段列表
            chunks: 检索到的候选文档块列表，每个至少包含
                    {'content': ..., 'file_name': ..., 'chunk_index': ...}
            model: 可选覆盖默认模型

        Returns:
            结构化记录列表，每条记录包含提取的字段 + 来源标注 + 置信度
        """
        if not chunks:
            return []

        # 组装候选文档
        docs_text = ""
        for idx, ch in enumerate(chunks):
            docs_text += (
                f"\n---\n"
                f"[Document {idx+1}: {ch.get('file_name', 'unknown')}, "
                f"chunk#{ch.get('chunk_index', '?')}]\n"
                f"{ch['content']}\n"
            )

        fields_desc = "\n".join(f"- {f}" for f in target_fields)

        system_prompt = f"""你是一个精确的数据提取专家。用户需要从候选文档中提取指定字段的数据。

注意：文档可能经过OCR识别，存在少量字符识别错误（如"染"→"柒"、"他"→"仟"、"圆整UNG"→"小写"等）。请根据上下文推断正确的文字和数字，不要因为OCR噪声就放弃提取。

## 用户问题
{user_query}

## 需要提取的字段
{fields_desc}
- 来源引用 (引用原文中支持该数据的句子)
- 置信度 (0.0-1.0 之间的小数)

## 重要规则
1. 以 JSON 数组格式输出，每个对象代表一条独立记录
2. OCR文本可能有噪声，请结合上下文推断真实值 — 例如「染拾玖万」应理解为「柒拾玖万」，「974000, 00」应理解为「974000.00」
3. 如果多条记录指向同一实体（如同一合同的不同 chunk），合并为一条
4. "来源引用" 必须引用原文中直接支持该数据的语句（即使有OCR噪声也要引用）
5. 置信度 = 1.0 表示字段值在原文中明确出现（即使有轻微OCR噪声），< 0.7 表示需要大量推断
6. 完全找不到相关信息的字段才填 null
7. 不要输出任何 JSON 之外的内容，不要加 ```json``` 包裹
8. 输出必须是合法的 JSON 数组"""

        model_name = model or self.cfg.llm_model

        message = self.anthropic.messages.create(
            model=model_name,
            max_tokens=self.cfg.llm_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": docs_text}],
        )

        # 解析响应
        raw = _get_text_from_response(message).strip()

        # 尝试提取 JSON
        return self._parse_json_response(raw)

    def extract_aggregate(
        self,
        user_query: str,
        chunks: list[dict],
        model: Optional[str] = None,
    ) -> dict:
        """聚合查询专用：从表格/文本中提取所有数值并计算汇总。

        与通用 extract 不同，本方法只提取数值列，不要求逐条结构化记录。
        """
        docs_text = ""
        for idx, ch in enumerate(chunks[:10]):
            docs_text += (
                f"\n---\n[Document {idx+1}: {ch.get('file_name', 'unknown')}, "
                f"chunk#{ch.get('chunk_index', '?')}]\n"
                f"{ch['content']}\n"
            )

        system_prompt = f"""你是精确的财务数据提取与汇总助手。用户问题：

{user_query}

从上述文档中提取所有合同金额/中标金额/成交金额数值，并计算总和。

## 规则
1. 只提取金额类数值（合同金额、中标金额、成交金额、总价等）
2. 金额单位可能是 元 / 万元 / 人民币大写，统一转为"元"
3. 输出格式（严格JSON，不要```json包裹）：
{{"values": [数值1, 数值2, ...], "total": 合计数值, "count": 数量, "note": "简要说明"}}
4. 只输出JSON，不要其他内容"""

        model_name = model or self.cfg.llm_model
        message = self.anthropic.messages.create(
            model=model_name,
            max_tokens=self.cfg.llm_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": docs_text}],
        )
        raw = _get_text_from_response(message).strip()
        # 提取JSON
        try:
            return json.loads(raw)
        except Exception:
            # 尝试从文本中截取 JSON
            import re as _re
            m = _re.search(r'\{.*\}', raw, _re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {}

    def infer_target_fields(self, user_query: str) -> list[str]:
        """从用户查询中推断需要提取的字段列表。

        例如 "Q4服务器采购花了多少钱，哪家供应商"
        → ["采购项目", "金额", "币种", "供应商名称", "日期"]
        """
        system = """你是一个数据需求分析师。根据用户的自然语言查询，推断需要从文档中提取哪些字段。
只输出一个 JSON 数组，数组元素是字段名称字符串。不要输出其他任何内容。"""

        message = self.anthropic.messages.create(
            model=self.cfg.llm_model,
            max_tokens=min(2048, self.cfg.llm_max_tokens),
            system=system,
            messages=[{"role": "user", "content": user_query}],
        )

        raw = _get_text_from_response(message).strip()
        fields = self._parse_json_response(raw)
        if isinstance(fields, list) and all(isinstance(f, str) for f in fields):
            return fields
        # 回退：用几个通用字段
        return ["标题", "金额", "日期", "相关方", "摘要"]

    @staticmethod
    def _parse_json_response(raw: str) -> list | dict:
        """鲁棒地解析 LLM 返回的 JSON"""
        raw = raw.strip()
        # 去掉可能的 markdown 包裹
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试找到最外层 [...] 或 {...}
            for start, end in [("[", "]"), ("{", "}")]:
                si = raw.find(start)
                ei = raw.rfind(end)
                if si != -1 and ei != -1:
                    try:
                        return json.loads(raw[si:ei+1])
                    except json.JSONDecodeError:
                        pass
            return []
