from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class Chunking_Embedding_Manager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def chunking_docs(self, docs, chunking_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", " ", "\n", ""]
        )

        split_docs = text_splitter.split_documents(docs)
        print(f"{len(docs)} documents splitting into {len(split_docs)} chunks.")
        return split_docs

    def _load_model(self):
        try:
            print(f"Loading Sentence Transformer Model {self.model_name} for embedding.")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"An error occured while embedding in transformer: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model is not loaded.")

        print(f"Generating Embeddings for {len(texts)} texts.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise ValueError("Model is not loaded properly.")
        return self.model.get_sentence_embedding_dimension()
