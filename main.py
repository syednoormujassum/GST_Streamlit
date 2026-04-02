import os
from src.data_loader import pdf_loader
from src.chunking import Chunking_Embedding_Manager
from src.vectorstore import VectorStore
from src.search_retreiver import RAGRetriever,ChatGroq,load_dotenv,os



if __name__=='__main__':
    docs=pdf_loader()
    chunks=Chunking_Embedding_Manager().chunking_docs(docs)
    texts=[doc.page_content for doc in chunks]
    embeddings=Chunking_Embedding_Manager().generate_embeddings(texts)
    vector_store=VectorStore().add_docs(chunks,embeddings)
