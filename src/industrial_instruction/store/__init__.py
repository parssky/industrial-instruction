"""Vector store stage and the retrieval interface used by generate and bench."""

from industrial_instruction.store.faiss_store import FaissStore, SearchHit
from industrial_instruction.store.retrieval import (
    RetrievalError,
    get_retriever,
    register_retriever,
    retriever_backend,
)

__all__ = [
    "FaissStore",
    "RetrievalError",
    "SearchHit",
    "get_retriever",
    "register_retriever",
    "retriever_backend",
]
