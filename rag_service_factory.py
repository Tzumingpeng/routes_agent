from enum import Enum
from .rag_service import RAGService
from .chroma_rag_service import ChromaRAGService
# 未来还可以 import FaissRAGService / MilvusRAGService …

class RAGType(str, Enum):
    CHROMA = "chromadb"
    FAISS  = "faiss"
    # …

class RAGServiceFactory:
    @staticmethod
    def get(kb_name: str, rag_type: RAGType, embed_model: str) -> RAGService:
        if rag_type == RAGType.CHROMA:
            return ChromaRAGService(kb_name, embed_model)
        # elif rag_type == RAGType.FAISS: ...
        else:
            raise ValueError(f"Unknown RAG type: {rag_type}")