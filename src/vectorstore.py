import numpy as np
from sentence_transformers import SentenceTransformer # Embedding Model will be present in Sentence Transformer
import chromadb,pickle
from chromadb.config import Settings
import uuid,os
from typing import List, Tuple,Any,Dict
from sklearn.metrics.pairwise import cosine_similarity
from src.chunking import Chunking_Embedding_Manager
class VectorStore():
    """Manages document embedding into vectore store chroma db"""
    def __init__(self,collection_name:str="pdf_documents",persist_directory:str="../data/vectorstore"):
        """
        Initialize Vector Store
        Args: collection_name = Name of ChromaDB Collection
        persist_directory= Directory to persist vector store
        """
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)

            # collections
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"pdf files collection of vectors embeddings for RAG pipeline."}
            )
            print(f"\nVector db initialized.\nCollection Name: {self.collection_name}")
            print(f"\n Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error occurred for initializing Vector Store: {e}")
            raise

    def add_docs(self,documents:List[Any],embeddings:np.ndarray):
        """
        Adding documents and their embeddings to vector store
        Args:
        documents= list of langchain documents
        embeddings= corresponding embeddings for the documents
        """
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match the number of embeddings")
        
        print(f"\n Adding {len(documents)} documents to vector store")

        print("\n Preparing data for Vector Store chroma db.")
        
        #Preparing data for Vector Store
        ids=[]
        metadatas=[]
        documents_text=[]
        embedding_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            #Generate unique ID
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #Prepare Metadata
            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            #Document Content
            documents_text.append(doc.page_content)

            #embedding
            embedding_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embedding_list
            )
            print(f"\n Successfully added {len(documents)} documents and {len(embeddings)} embeddings to Vector Store")
            print(f"\n Total number of documents in collection are {self.collection.count()}")

        except Exception as e:
            print(f"\n An error occurred while adding documents to Vector Store: {e}")
            raise


# Initializing vector store
vector_store=VectorStore()