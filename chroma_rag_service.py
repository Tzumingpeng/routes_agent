import chromadb
from langchain_openai import ChatOpenAI

from langchain.docstore.document import Document

from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config_env import OPENAI_API_KEY, OPENAI_API_URL
from .rag_service import RAGService

class ChromaRAGService(RAGService):
    def _init_store(self):
        self.client = chromadb.PersistentClient(path=f"./{self.kb_name}_vs")
        try:
            self.col = self.client.get_collection(self.kb_name)
        except ValueError:
            self.col = self.client.create_collection(self.kb_name)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def _add_docs(self, docs):
        chunks = []
        for d in docs:
            chunks.extend(self.splitter.create_documents([d.page_content]))
        texts  = [c.page_content for c in chunks]
        self.col.add(documents=texts)      # 省略元数据/嵌入细节
        return len(texts)

    def _search(self, query, k, thr):
        res = self.col.query(query_texts=[query], n_results=k)
        pairs = list(zip(res["documents"][0], res["distances"][0]))
        return [(Document(page_content=t), d) for t, d in pairs if d <= thr]

    def _llm_generate(self, query, context):
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_URL,
            temperature=0.3,
        )
        prompt = f"已知资料：\n{context}\n\n回答问题：{query}"
        return llm.invoke([HumanMessage(content=prompt)]).content