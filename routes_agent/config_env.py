# routes_agent/config_env.py (保持原样，无需修改)
"""加载 .env，并集中存放所有全局常量。"""
from dotenv import load_dotenv
import os

# === 环境变量 ===
load_dotenv(override=True)          # 读取 .env
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL       = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")

# === 运行时开关 ===
# 本地是否装了 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    USE_ST = True
except ImportError:
    USE_ST = False