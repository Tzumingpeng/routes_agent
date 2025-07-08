"""演示：调用 RAG 推荐、Google 路线并输出整体行程。"""
from rag_system import TravelRAGSystem
from tools import rag_recommend_attractions, google_city_matrix
from config_env import OPENAI_API_KEY   # 确保 .env 已配置

rag = TravelRAGSystem()
if rag.collection.count() == 0:
    print("⚠️ 知识库为空，可先调用 rag.add_documents(...) 添加默认 or 用户数据。")

def plan_trip(user_prompt: str):
    # 抽取城市（简化：直接 split）
    cities = [c.strip() for c in user_prompt.replace("、", ",").split(",") if c.strip()]
    print("🏖️ 目标城市:", cities)

    rec = rag_recommend_attractions(rag, cities)
    print("\n🎯 RAG 景点推荐:\n", rec)

    if len(cities) > 1:
        matrix = google_city_matrix(cities)
        print("\n🛣️ 城市间路线:\n", matrix)

if __name__ == "__main__":
    plan_trip("我想去台北、花莲和高雄，从台北出发")
