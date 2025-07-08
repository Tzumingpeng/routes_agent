from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain.docstore.document import Document

class RAGService(ABC):
    """检索-增强-生成统一接口"""

    def __init__(self, kb_name: str, embed_model: str):
        self.kb_name  = kb_name
        self.embed_model = embed_model
        self._init_store()          

    # ---------- 统一对外 API ----------
    def add_docs(self, docs: List[Document]):
        return self._add_docs(docs)

    def search(self, query: str, k: int, thr: float):
        return self._search(query, k, thr)

    def answer(self, query: str, k: int = 5, thr: float = 0.5) -> str:
        """完整 RAG 流程：检索 → 组 prompt → LLM 生成"""
        hits = self.search(query, k, thr)
        context = "\n\n".join([d.page_content for d, _ in hits])
        return self._llm_generate(query, context)

    # ---------- 子类实现 ----------
    @abstractmethod
    def _init_store(self): ...
    @abstractmethod
    def _add_docs(self, docs: List[Document]): ...
    @abstractmethod
    def _search(self, query: str, k: int, thr: float) -> List[Tuple[Document, float]]: ...
    @abstractmethod
    def _llm_generate(self, query: str, context: str) -> str: ...