from typing import Any
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json


class Retriever:
    def __init__(self, index_path: str= "./faiss_index/index",
                 mapping_path: str = "./faiss_index/id_map.json") -> None:

        # self.embed_model = AutoModel.from_pretrained(
        #     "./models/tooka"
        # )
        self.embed_model = SentenceTransformer("../models/embeddinggemma-300m", device='cuda')

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

    def add_embedding(self, embedding, original_doc: str):
        emb = self._normalize(embedding)
        doc_id = self.next_id

        self.index.add(emb.reshape(1, -1))
        self.id_to_doc[doc_id] = original_doc

        self.next_id += 1
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

    def fast_add_documents(self, docs: list[str], batch_size: int = 64):
        """
        Fast batch addition of documents.
        Does not modify existing 'add_document' or search logic.
        """
        all_embeddings = []
        start_id = self.next_id

        # --- encode in batches ---
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            emb = self.embed_model.encode(batch, batch_size=batch_size, show_progress_bar=False)
            emb = self._normalize(emb)
            all_embeddings.append(emb)

        all_embeddings = np.vstack(all_embeddings)

        # --- add all embeddings to FAISS at once ---
        self.index.add(all_embeddings)

        # --- update id_to_doc mapping ---
        for i, doc in enumerate(docs):
            self.id_to_doc[start_id + i] = doc

        self.next_id += len(docs)

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
    
    def clean_empty_documents(self):
        """
        Removes empty documents from id_to_doc and rebuilds the FAISS index.
        Useful if index contains "" values or corrupted entries.
        """
        print("[INFO] Cleaning empty documents...")

        # Keep only non-empty documents
        new_id_to_doc = {}
        valid_docs = []
        for doc_id, text in self.id_to_doc.items():
            if isinstance(text, str) and text.strip() != "" and len(text.split()) > 10:
                new_id_to_doc[len(new_id_to_doc)] = text
                valid_docs.append(text)

        print(f"[INFO] Original documents: {len(self.id_to_doc)}")
        print(f"[INFO] Cleaned documents: {len(new_id_to_doc)}")

        # Rebuild FAISS index
        emb_dim = self.embed_model.get_sentence_embedding_dimension()
        new_index = faiss.IndexFlatIP(emb_dim)

        # Re-encode all cleaned docs
        if len(valid_docs) > 0:
            embeddings = self.embed_model.encode(valid_docs, batch_size=64, show_progress_bar=True)
            embeddings = self._normalize(embeddings)
            new_index.add(embeddings)

        # Replace old index + mapping
        self.id_to_doc = new_id_to_doc
        self.index = new_index
        self.next_id = len(self.id_to_doc)

        print("[INFO] Cleanup complete.")    