# routes_agent/rag_system.py (修复collection创建问题)
"""TravelRAGSystem 负责：加载知识、生成 / 查询嵌入、维护 Chroma collection。"""
from typing import List
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 修复：使用绝对导入
try:
    from config_env import USE_ST, OPENAI_API_KEY, OPENAI_API_URL
except ImportError:
    from .config_env import USE_ST, OPENAI_API_KEY, OPENAI_API_URL

if USE_ST:
    from sentence_transformers import SentenceTransformer
else:
    from openai import OpenAI

class TravelRAGSystem:
    def __init__(self, persist_dir: str = "./travel_vectordb"):
        self.persist_dir = persist_dir
        self.client      = chromadb.PersistentClient(path=persist_dir)
        self.splitter    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedder    = (SentenceTransformer('all-MiniLM-L6-v2') if USE_ST else None)
        
        # 修复：确保collection存在
        self.collection = self._ensure_collection("travel_knowledge")
        
        # 检查知识库状态
        try:
            count = self.collection.count()
            print(f"知识库包含 {count} 条记录")
        except Exception as e:
            print(f"检查知识库时出错: {e}")

    def _ensure_collection(self, name: str):
        """确保collection存在，如果不存在就创建"""
        try:
            # 尝试获取现有collection
            collection = self.client.get_collection(name)
            print(f"✅ 找到现有collection: {name}")
            return collection
        except ValueError:
            # collection不存在，创建新的
            print(f"📦 创建新collection: {name}")
            return self.client.create_collection(name)
        except Exception as e:
            print(f"❌ collection操作失败: {e}")
            # 强制创建新collection
            try:
                return self.client.create_collection(name)
            except Exception as e2:
                print(f"❌ 创建collection失败: {e2}")
                raise e2

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入"""
        if USE_ST and self.embedder:
            return self.embedder.encode(texts).tolist()
        
        # 使用 OpenAI 嵌入
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_URL)
            embeddings = []
            for text in texts:
                response = client.embeddings.create(
                    model="text-embedding-ada-002", 
                    input=text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            print(f"嵌入生成失败: {e}")
            # 返回零向量作为fallback
            return [[0.0] * 1536] * len(texts)

    def add_documents(self, docs: List[dict]):
        """添加文档到知识库"""
        for i, doc in enumerate(docs):
            chunks = self.splitter.split_text(doc["content"])
            if chunks:  # 确保有内容
                try:
                    embeddings = self._embed(chunks)
                    for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        self.collection.add(
                            documents=[chunk],
                            metadatas=[doc.get("metadata", {})],
                            ids=[f"doc_{i}_{j}"],
                            embeddings=[embedding],
                        )
                except Exception as e:
                    print(f"添加文档失败: {e}")

    def query(self, text: str, k: int = 5) -> List[str]:
        """查询相关文档"""
        try:
            res = self.collection.query(query_texts=[text], n_results=k)
            return res["documents"][0] if res["documents"] else []
        except Exception as e:
            print(f"查询失败: {e}")
            return []