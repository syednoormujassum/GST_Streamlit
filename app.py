import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.chunking_new import Chunking_Embedding_Manager
from src.data_loader import pdf_loader, list_pdf_files
from src.vectorstore import VectorStore
from src.search_retreiver import RAGRetriever

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"

app = FastAPI(
    title="GST Knowledge Copilot",
    description="A lightweight FastAPI UI for GST retrieval-augmented generation.",
    version="1.0.0"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

vector_store = VectorStore(persist_directory=BASE_DIR / "data" / "vectorstore")
embedding_manager = Chunking_Embedding_Manager()
rag_retriever = RAGRetriever(vector_store, embedding_manager)


def _compare_pdf_metadata(current_files: list[dict], saved_metadata: dict) -> bool:
    saved_files = saved_metadata.get("files", []) if saved_metadata else []
    if len(current_files) != len(saved_files):
        return True

    saved_map = {item["path"]: item for item in saved_files}
    for pdf_file in current_files:
        saved_file = saved_map.get(pdf_file["path"])
        if not saved_file:
            return True
        if saved_file.get("mtime") != pdf_file.get("mtime") or saved_file.get("size") != pdf_file.get("size"):
            return True
    return False


@app.get("/api/status")
def app_status():
    current_files = list_pdf_files(str(DATA_PATH))
    saved_meta = vector_store.load_index_metadata()
    saved_index_exists = vector_store.has_saved_store()
    new_data_detected = False

    if saved_index_exists and saved_meta is not None:
        new_data_detected = _compare_pdf_metadata(current_files, saved_meta)
    elif saved_index_exists and saved_meta is None:
        new_data_detected = bool(current_files)
    elif not saved_index_exists and current_files:
        new_data_detected = True

    return {
        "status": "ok",
        "saved_index_exists": saved_index_exists,
        "documents_in_store": vector_store.collection.count(),
        "pdf_count": len(current_files),
        "new_data_detected": new_data_detected,
        "current_files": current_files,
    }


@app.post("/api/index")
async def manage_index(action: str = Form(...)):
    action = action.strip().lower()

    if action not in {"use_saved", "rebuild"}:
        return JSONResponse({"error": "Action must be either 'use_saved' or 'rebuild'."}, status_code=400)

    if action == "use_saved":
        if not vector_store.has_saved_store():
            return JSONResponse({"error": "No saved vector store exists. Please rebuild first."}, status_code=400)
        return {
            "status": "loaded",
            "message": "Using existing saved vector store.",
            "documents_in_store": vector_store.collection.count(),
        }

    docs = pdf_loader(str(DATA_PATH))
    if not docs:
        return JSONResponse({"error": "No PDF documents found to build the index."}, status_code=400)

    pdf_files = list_pdf_files(str(DATA_PATH))
    vector_store.clear_store()
    chunks = embedding_manager.chunking_docs(docs)
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_docs(chunks, embeddings)
    vector_store.save_index_metadata(pdf_files)

    return {
        "status": "rebuilt",
        "message": "Vector store rebuilt successfully.",
        "documents_in_store": vector_store.collection.count(),
        "pdf_count": len(pdf_files),
    }


@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def query(question: str = Form(...)):
    question = question.strip()
    if not question:
        return JSONResponse({"error": "Query text cannot be empty."}, status_code=400)

    response = rag_retriever.Advanced_RAG(question)
    return response


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "documents_in_store": vector_store.collection.count()
    }
