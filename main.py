from pathlib import Path
from dotenv import load_dotenv
from src.data_loader import pdf_loader, list_pdf_files
from src.chunking_new import Chunking_Embedding_Manager
from src.vectorstore import VectorStore

load_dotenv()

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data"
    persist_path = project_root / "data" / "vectorstore"

    docs = pdf_loader(str(data_path))
    pdf_files = list_pdf_files(str(data_path))
    embedding_manager = Chunking_Embedding_Manager()
    chunks = embedding_manager.chunking_docs(docs)
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    vector_store = VectorStore(persist_directory=persist_path)
    vector_store.add_docs(chunks, embeddings)
    vector_store.save_index_metadata(pdf_files)

    print("Document processing completed. Vector store is ready.")
