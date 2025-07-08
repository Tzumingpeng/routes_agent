# routes_agent/tools.py (修复导入)
"""把所有工具函数集中放在这里，方便在别处复用。"""
import requests, itertools, json
from typing import List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 修复：使用绝对导入
try:
    from config_env import GOOGLE_MAPS_API_KEY, OPENAI_API_KEY, OPENAI_API_URL
    from rag_system import TravelRAGSystem
except ImportError:
    from .config_env import GOOGLE_MAPS_API_KEY, OPENAI_API_KEY, OPENAI_API_URL
    from .rag_system import TravelRAGSystem

# ---------- Google Maps ----------
def google_route(origin: str, dest: str, mode: str = "DRIVE") -> str:
    if not GOOGLE_MAPS_API_KEY:
        return "❌ Google Maps API 密钥未配置"

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
    }
    body = {
        "origin": {"address": origin},
        "destination": {"address": dest},
        "travelMode": mode,
        "languageCode": "zh-TW",
        "units": "METRIC",
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=20)
        if not r.ok:
            return f"{origin} → {dest}: 查询失败({r.status_code})"

        data = r.json()
        if not data.get("routes"):
            return f"{origin} → {dest}: 未找到路线"
            
        route = data["routes"][0]

        # --------- 兼容两种 duration 结构 ---------
        dur = route.get("duration", 0)
        if isinstance(dur, dict):                  # {"seconds": 5321}
            seconds = int(dur.get("seconds", 0))
        elif isinstance(dur, str) and dur.endswith("s"):   # "5321s"
            seconds = int(dur[:-1])
        else:
            seconds = 0
        hours, minutes = divmod(seconds // 60, 60)
        dur_txt = f"{hours}h{minutes}m" if hours else f"{minutes}m"
        # ----------------------------------------

        km = route.get("distanceMeters", 0) / 1000
        return f"{origin} → {dest}: {km:.1f} km, {dur_txt}"
    except Exception as e:
        return f"{origin} → {dest}: 查询出错 - {str(e)}"


def google_city_matrix(cities: List[str], mode="DRIVE") -> str:
    return "\n".join(
        google_route(a, b, mode) for a, b in itertools.permutations(cities, 2)
    )

# ---------- RAG 景点推荐工具 ----------
def rag_recommend_attractions(rag: TravelRAGSystem, cities: List[str]) -> str:
    # 修复：正确聚合上下文
    context_chunks = []
    for city in cities:
        hits = rag.query(f"{city} 景点 交通 住宿", k=3)
        context_chunks.extend(hits)   # 平铺到同一个 list
    context = "\n\n".join(context_chunks) if context_chunks else "暂无知识库信息"

    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_URL,
            temperature=0.3,
        )
        
        cities_str = "、".join(cities)
        prompt = f"""基于以下知识库信息，为指定城市推荐景点：

知识库信息：
{context}

目标城市：{cities_str}

请按以下格式回答：
城市名：
1. 景点名称 - 详细描述
2. 景点名称 - 详细描述

要求：
- 优先使用知识库中的信息
- 每个城市推荐2-3个景点
- 如果知识库没有信息，基于常识推荐"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"推荐失败：{str(e)}"

# ---------- 辅助函数 ----------
def extract_cities_from_text(text: str) -> List[str]:
    """简单的城市提取"""
    # 使用正则表达式提取中文城市名
    import re
    cities = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
    # 过滤掉常见非城市词汇
    filtered = [c for c in cities if c not in ['我想', '出发', '旅行', '景点', '推荐', '规划']]
    return filtered[:5]  # 最多5个城市