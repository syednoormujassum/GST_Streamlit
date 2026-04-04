import gc
import json
import shutil
import numpy as np
import chromadb
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import List, Any


class VectorStore:
    """Manages document embedding into vector store (ChromaDB)."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str | Path | None = None):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory) if persist_directory is not None else Path(__file__).resolve().parent.parent / "data" / "vectorstore"
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "pdf files collection of vector embeddings for RAG pipeline."}
            )
            print(f"\nVector db initialized.\nCollection Name: {self.collection_name}")
            print(f"\nExisting documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error occurred for initializing Vector Store: {e}")
            raise

    @property
    def metadata_path(self) -> Path:
        return self.persist_directory / "index_meta.json"

    def has_saved_store(self) -> bool:
        return self.persist_directory.exists() and self.collection is not None and self.collection.count() > 0

    def clear_store(self):
        if self.collection is not None:
            try:
                self.collection.delete()
            except Exception:
                pass

        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

        self.client = None
        self.collection = None

        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

        gc.collect()
        self._initialize_store()

    def save_index_metadata(self, pdf_files: list[dict]):
        metadata = {
            "files": pdf_files,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "collection_name": self.collection_name,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load_index_metadata(self) -> dict | None:
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def add_docs(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")

        print(f"\nAdding {len(documents)} documents to vector store")
        print("\nPreparing data for Vector Store chroma db.")

        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embedding_list
            )
            print(f"\nSuccessfully added {len(documents)} documents and {len(embeddings)} embeddings to Vector Store")
            print(f"\nTotal number of documents in collection are {self.collection.count()}")
        except Exception as e:
            print(f"\nAn error occurred while adding documents to Vector Store: {e}")
            raise
