# routes_agent/main.py (使用原有的extract_cities_from_prompt函数)
"""演示：调用 RAG 推荐、Google 路线并输出整体行程。"""
import sys
import os
import requests  # 添加requests导入
import json

# 修复：支持不同的运行方式
try:
    # 当作为模块运行时 (python -m routes_agent.main)
    from routes_agent.rag_system import TravelRAGSystem
    from routes_agent.tools import rag_recommend_attractions, google_city_matrix
    from routes_agent.config_env import OPENAI_API_KEY, GOOGLE_MAPS_API_KEY
except ImportError:
    # 当直接运行时 (python main.py)
    from rag_system import TravelRAGSystem
    from tools import rag_recommend_attractions, google_city_matrix
    from config_env import OPENAI_API_KEY, GOOGLE_MAPS_API_KEY

# 导入你原有的城市提取函数
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 导入你原版中的辅助函数
def extract_attractions_from_recommendations(recommendations: str) -> str:
    """从景点推荐结果中提取景点名称"""
    attractions = []
    
    lines = recommendations.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 查找编号开头的行 (如 "1. 台北101 - 描述")
        if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
            # 提取景点名称
            parts = line.split(' - ')
            if len(parts) >= 2:
                # 去掉编号，提取景点名称
                attraction_name = parts[0].split('. ', 1)[-1].strip()
                attractions.append(attraction_name)
            else:
                # 如果没有 " - " 分隔符，尝试其他方式
                cleaned = line.split('. ', 1)[-1].strip()
                if cleaned:
                    first_word = cleaned.split()[0] if cleaned.split() else cleaned
                    attractions.append(first_word)
    
    # 去重并返回逗号分隔的字符串
    unique_attractions = list(set(attractions))
    return ','.join(unique_attractions)

def get_attraction_hours(attractions_str: str) -> str:
    """获取景点的营业时间信息 - 使用你原版的实现"""
    try:
        from config_env import GOOGLE_MAPS_API_KEY
    except ImportError:
        from .config_env import GOOGLE_MAPS_API_KEY
        
    if not GOOGLE_MAPS_API_KEY:
        return "Google Maps API密钥未配置"
    
    try:
        attractions = [attr.strip() for attr in attractions_str.split(',')]
        results = []
        
        for attraction in attractions:
            # 使用新版Places API搜索景点
            search_url = "https://places.googleapis.com/v1/places:searchText"
            
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.businessStatus,places.currentOpeningHours,places.regularOpeningHours"
            }
            
            request_body = {
                "textQuery": attraction,
                "languageCode": "en",
                "regionCode": "TW",
                "includedType": "tourist_attraction"
            }
            
            search_response = requests.post(search_url, headers=headers, json=request_body, timeout=15)
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                
                if "places" in search_data and len(search_data["places"]) > 0:
                    place = search_data["places"][0]
                    
                    # 提取信息
                    name = place.get("displayName", {}).get("text", attraction)
                    address = place.get("formattedAddress", "地址未知")
                    rating = place.get("rating", "无评分")
                    rating_count = place.get("userRatingCount", 0)
                    business_status = place.get("businessStatus", "未知")
                    
                    # 提取营业时间
                    current_hours = place.get("currentOpeningHours", {})
                    regular_hours = place.get("regularOpeningHours", {})
                    
                    # 优先使用当前营业时间，如果没有则使用常规营业时间
                    hours_info = current_hours or regular_hours
                    
                    if hours_info and "weekdayDescriptions" in hours_info:
                        hours_text = "\n".join(hours_info["weekdayDescriptions"])
                        current_status = "营业中" if hours_info.get("openNow", False) else "未营业"
                    else:
                        hours_text = "营业时间未知 (可能为24小时开放的户外景点)"
                        current_status = "状态未知"
                    
                    result = f"""景点：{name}
地址：{address}
评分：{rating} ({rating_count} reviews)
营业状态：{business_status}
当前状态：{current_status}
营业时间：
{hours_text}"""
                    
                    results.append(result)
                else:
                    results.append(f"景点：{attraction}\n搜索返回空结果")
            else:
                results.append(f"景点：{attraction}\nAPI调用失败 (HTTP {search_response.status_code})")
                
        return "\n\n" + "="*50 + "\n\n".join(results)
        
    except Exception as e:
        return f"营业时间查询错误：{str(e)}"

def extract_cities_from_prompt(prompt: str) -> list[str]:
    """从用户提示中提取城市名称 - 使用你原有的函数"""
    try:
        # 使用LLM来提取城市名称
        # 导入配置
        try:
            from config_env import OPENAI_API_URL
        except ImportError:
            from .config_env import OPENAI_API_URL
            
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_base=OPENAI_API_URL,  # 修复：使用正确的URL变量
            openai_api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        
        extraction_prompt = f"""请从以下用户提示中提取所有城市名称：

用户提示："{prompt}"

要求：
1. 只提取城市名称，不要其他信息
2. 以JSON格式返回，例如：["台北", "花莲", "高雄"]
3. 如果没有找到城市，返回空数组[]
4. 确保城市名称准确无误"""

        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        
        # 尝试解析JSON响应
        try:
            cities = json.loads(response.content)
            if isinstance(cities, list):
                return cities
        except:
            # 如果JSON解析失败，尝试简单的文本解析
            import re
            # 从回复中提取引号内的内容
            quoted_cities = re.findall(r'"([^"]+)"', response.content)
            if quoted_cities:
                return quoted_cities
        
        return []
        
    except Exception as e:
        print(f"城市提取失败: {e}")
        return []

def test_system():
    """测试系统功能"""
    print("🧪 测试系统功能...")
    
    # 测试RAG系统
    try:
        rag = TravelRAGSystem()
        count = rag.collection.count()
        print(f"✅ RAG系统正常，知识库有 {count} 条记录")
        
        # 测试查询
        results = rag.query("台北景点", k=2)
        if results:
            print(f"✅ 查询功能正常")
        else:
            print("⚠️ 查询结果为空")
        
        return rag
    except Exception as e:
        print(f"❌ RAG系统错误: {e}")
        return None

def plan_trip(user_prompt: str, rag: TravelRAGSystem):
    """完整的RAG增强智能旅行规划"""
    print(f"\n🎯 用户需求: {user_prompt}")
    print("-" * 50)
    
    # 步骤1: 提取城市
    cities = extract_cities_from_prompt(user_prompt)
    if not cities:
        print("❌ 未识别到城市名称")
        return
    
    print(f"🏙️ 识别城市: {', '.join(cities)}")
    
    # 步骤2: RAG增强景点推荐
    print(f"\n📍 步骤1: RAG增强景点推荐")
    print("-" * 40)
    
    try:
        recommendations = rag_recommend_attractions(rag, cities)
        print("🤖 基于知识库的景点推荐:")
        print(recommendations)
        
        # 步骤3: 从RAG推荐中提取景点名称
        print("\n🔍 步骤2: 提取推荐的景点名称")
        print("-" * 40)
        
        attractions = extract_attractions_from_recommendations(recommendations)
        attraction_list = [attr.strip() for attr in attractions.split(',') if attr.strip()]
        
        if attraction_list:
            print(f"📋 提取到的景点: {', '.join(attraction_list)}")
            
            # 步骤4: 调用Google Places API获取详细信息
            print("\n⏰ 步骤3: 获取景点实时营业时间和详细信息")
            print("-" * 40)
            
            attraction_details = get_attraction_hours(','.join(attraction_list))
            print("📊 景点详细信息:")
            print(attraction_details)
            
        else:
            print("⚠️ 未能从推荐中提取到具体景点名称")
            attraction_details = "无法获取景点详细信息"
        
        # 步骤5: 调用Google Maps API规划路线
        city_routes = ""
        if len(cities) > 1:
            print("\n🛣️ 步骤4: 城市间路线规划")
            print("-" * 40)
            print("🔄 正在查询路线信息...")
            
            try:
                city_routes = google_city_matrix(cities)
                print("🗺️ 城市间路线:")
                print(city_routes)
            except Exception as e:
                print(f"❌ 路线查询失败: {e}")
                city_routes = "路线查询失败"
        else:
            city_routes = "单个城市，无需城市间路线规划"
            print("\n🛣️ 单个城市，无需城市间路线规划")
        
        # 步骤6: 综合生成完整旅行规划
        print("\n📋 步骤5: 综合旅行规划")
        print("-" * 40)
        
        try:
            # 导入配置
            try:
                from config_env import OPENAI_API_URL
            except ImportError:
                from .config_env import OPENAI_API_URL
                
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_base=OPENAI_API_URL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0.3
            )
            
            planning_prompt = f"""基于以下信息，制定详细的旅行规划：

用户需求：{user_prompt}
目标城市：{', '.join(cities)}

RAG景点推荐：
{recommendations}

景点详细信息（营业时间等）：
{attraction_details}

城市间路线：
{city_routes}

请制定一个详细的旅行计划，包括：
1. 推荐的游览顺序
2. 每个景点的最佳游览时间
3. 基于营业时间的时间安排
4. 交通建议
5. 具体时间安排，精确到时间点

要求：
- 充分利用RAG推荐的景点信息
- 考虑实际的营业时间安排
- 提供可执行的具体建议"""

            planning_response = llm.invoke([HumanMessage(content=planning_prompt)])
            
            print("🎯 最终旅行规划:")
            print("=" * 60)
            print(planning_response.content)
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 综合规划生成失败: {e}")
            
    except Exception as e:
        print(f"❌ 景点推荐失败: {e}")
    
    print("=" * 50)

def main():
    print("🚀 启动旅行规划系统")
    
    # 检查配置
    if not OPENAI_API_KEY:
        print("❌ 请在 .env 文件中配置 OPENAI_API_KEY")
        
    # 测试系统
    rag = test_system()
    if not rag:
        print("❌ 系统初始化失败")
        return
    
    # 直接进入交互模式，跳过演示案例
    print("\n💬 交互模式 (输入 'quit' 退出):")
    
    while True:
        try:
            user_input = input("\n请输入旅行需求: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            if user_input:
                plan_trip(user_input, rag)
        except KeyboardInterrupt:
            break
    
    print("👋 再见！")

if __name__ == "__main__":
    main()