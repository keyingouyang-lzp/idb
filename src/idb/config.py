"""配置管理 — 从环境变量和 .env 文件加载设置"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 尝试加载 .env
try:
    from dotenv import load_dotenv
    _ENV_FILE = Path.home() / ".idb" / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
    load_dotenv()  # 也加载当前目录的 .env
except ImportError:
    pass


@dataclass
class Config:
    """全局配置单例"""

    # ---- 存储路径 ----
    db_path: str = field(default_factory=lambda: str(
        Path(os.getenv("IDB_DB_PATH", Path.home() / "idb_data"))
    ))

    # ---- Embedding ----
    embedding_model: str = field(default_factory=lambda:
        os.getenv("IDB_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    embedding_dim: int = 1024  # BGE-M3 默认; OpenAI small=1536 会自动适配
    embedding_provider: str = field(default_factory=lambda:
        os.getenv("IDB_EMBEDDING_PROVIDER", "openai")
    )

    # ---- OpenAI (Embedding + 可选LLM) ----
    openai_api_key: Optional[str] = field(default_factory=lambda:
        os.getenv("OPENAI_API_KEY")
    )
    openai_base_url: Optional[str] = field(default_factory=lambda:
        os.getenv("OPENAI_BASE_URL")
    )

    # ---- Anthropic (LLM 提取) ----
    anthropic_api_key: Optional[str] = field(default_factory=lambda:
        os.getenv("ANTHROPIC_API_KEY")
    )
    anthropic_base_url: Optional[str] = field(default_factory=lambda:
        os.getenv("ANTHROPIC_BASE_URL")
    )

    # ---- LLM 提取 ----
    llm_model: str = field(default_factory=lambda:
        os.getenv("IDB_LLM_MODEL", "claude-sonnet-4-6")
    )
    llm_provider: str = field(default_factory=lambda:
        os.getenv("IDB_LLM_PROVIDER", "anthropic")
    )
    # 推理型模型（如 deepseek-v4-pro）会先输出 thinking 块再输出正文，
    # 4096 token 的预算会被思考内容吃满导致空响应，故默认放宽。
    llm_max_tokens: int = field(default_factory=lambda:
        int(os.getenv("IDB_LLM_MAX_TOKENS", "16384"))
    )

    # 财务关键词提权：在真实语料上它与 RRF 的名次间距量级不匹配
    # （+0.008 vs 相邻名次 ~0.0003），会把相关 chunk 挤出 top-5，
    # 实测端到端成功率不变而 MRR@5 由 0.857 降到 0.762，故默认关闭。
    financial_boost: bool = field(default_factory=lambda:
        os.getenv("IDB_FINANCIAL_BOOST", "0") == "1"
    )

    # ---- 分块参数 ----
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # ---- 检索参数 ----
    top_k_retrieve: int = 30
    top_k_rerank: int = 30

    def ensure_api_keys(self):
        """确保必要的 API key 已设置"""
        missing = []
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY (用于 LLM 提取)")
        if missing:
            raise RuntimeError(
                f"缺少 API Key: {', '.join(missing)}\n"
                f"请设置环境变量或创建 ~/.idb/.env 文件\n"
                f"(Embedding 将使用本地模型 BGE-small-zh，无需 API key)"
            )


# 全局单例
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def init_config(db_path: Optional[str] = None, **kwargs):
    """用自定义参数初始化配置"""
    global _config
    cfg = Config()
    if db_path:
        cfg.db_path = db_path
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    _config = cfg
    return cfg
