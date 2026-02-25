from typing import Any
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json


class Retriever:
    def __init__(self, index_path: str= "./faiss_index/index",
                 mapping_path: str = "./faiss_index/id_map.json", embed_model_path: str = "../models/embeddinggemma-300m") -> None:

        # self.embed_model = AutoModel.from_pretrained(
        #     "./models/tooka"
        # )
        self.embed_model = SentenceTransformer(embed_model_path, device='cpu')

        self.index_path = index_path
        self.mapping_path = mapping_path

        # -----------------------------
        # Load stored docs (task objects)
        # -----------------------------
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r") as f:
                self.id_to_doc = json.load(f)
            self.id_to_doc = {int(k): v for k, v in self.id_to_doc.items()}
            self.next_id = max(self.id_to_doc.keys()) + 1
        else:
            self.id_to_doc = {}
            self.next_id = 0

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"Loaded index successfully! {index_path}")
        else:
            # emb_dim = self.embed_model.config.hidden_size # Jina version
            emb_dim = self.embed_model.get_sentence_embedding_dimension() # Sentece Transformers Version
            self.index = faiss.IndexFlatIP(emb_dim)

    # -----------------------------
    # Encode text using Jina-v3
    # -----------------------------
    def _encode(self, query: str):
        # emb = self.embed_model.encode(query, task='text-matching') # Jina model
        emb = self.embed_model.encode(query)
        return np.array(emb)

    # -----------------------------
    # Normalize for cosine similarity
    # -----------------------------
    def _normalize(self, vec):
        vec = np.array(vec, dtype=np.float32) # (seq_len, embdding_size)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        return (vec / (norm + 1e-9)).astype("float32")

    # -----------------------------
    # Add a NEW task (with no ID)
    # doc must be a dict with task fields
    # -----------------------------
    def add_document(self, doc: str):
        # Store single task only
        doc_id = self.next_id
        
        self.id_to_doc[doc_id] = doc

        # Index text for retrieval
        emb = self._encode(doc)
        emb = self._normalize(emb)
        self.index.add(emb)

        self.next_id += 1


    def search(self, query: str, k: int = 5):
        q = self._encode(query)
        q = self._normalize(q)
        distances, indices = self.index.search(q, k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue

            obj = self.id_to_doc[idx]
            results.append(obj)   

        return results


    def get_docs(self):
        return list(self.id_to_doc.values())

    # -----------------------------
    # Update existing tasks
    # input: [{"id": "3", ...updated task fields...}, ...]
    # -----------------------------
    def update_docs(self, updated_docs: list[dict]):
        for new_task in updated_docs:
            doc_id = int(new_task["id"])
            if doc_id not in self.id_to_doc.keys():
                print(f"[WARNNING] Model has generated Id by itself :{doc_id}")
                new_task.pop("id")
                self.add_document(new_task)
                continue
            self.id_to_doc[doc_id] = new_task

    # -----------------------------
    # Save both FAISS + mapping
    # -----------------------------
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "w") as f:
            json.dump(self.id_to_doc, f, ensure_ascii=False, indent=2)