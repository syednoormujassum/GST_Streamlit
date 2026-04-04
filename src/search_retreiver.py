from typing import List, Any, Dict
from langchain_groq import ChatGroq
import os


class RAGRetriever:
    """Handles query based retrieval from vector store."""

    def __init__(self, vector_store, embedding_manager, groq_model: str = "llama-3.1-8b-instant"):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API"),
            temperature=0.1,
            model=os.getenv("GROQ_MODEL", groq_model),
            max_tokens=1024
        )

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: {query}")
        print(f"top_k: {top_k}, score_threshold={score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similarity_score,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} docs after filtering")
            else:
                print("No documents found for retrieval based on similarity search")
            return retrieved_docs
        except Exception as e:
            print(f"Error found during retrieval: {e}")
            return []

    def Advanced_RAG(self, query: str, top_k: int = 5, min_score: float = 0.2):
        results = self.retrieve(query, top_k=top_k, score_threshold=min_score)

        if not results:
            return {
                "Answer": "No relevant context found",
                "Sources": [],
                "Confidence_Score": 0.0,
                "Context": ""
            }

        context = "\n\n".join([doc["content"] for doc in results])
        sources = [
            {
                "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:500] + "......",
            }
            for doc in results
        ]
        confidence = max(doc["similarity_score"] for doc in results)

        prompt = f"""
            Use the following context and answer the query concisely.

            Context: {context}
            Question: {query}
            Answer:
        """

        response = self.llm.invoke([prompt])
        output = {
            "Answer": response.content,
            "Sources": sources,
            "Confidence_Score": confidence,
            "Context": context
        }
        return output
