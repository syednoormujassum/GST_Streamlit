from src.chunking import Chunking_Embedding_Manager
from src.vectorstore import VectorStore
from typing import List,Any,Dict
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

class RAGRetriever:
      '''
        Handles query based retireval from vector store.
        '''
      def __init__(self,vector_store:VectorStore,embedding_manager:Chunking_Embedding_Manager):
        """
        Initialize the Retriever..
        Args: 
        vector_store: Vector store containing document embeddings
        embedding_manager: Manager for generating query embeddings
        """
        self.vector_store=vector_store
        self.embedding_manager=embedding_manager
        #GROQ_API=os.getenv('OPENGROQ_API_KEY')
        self.llm=(ChatGroq(groq_api_key=os.getenv("GROQ_API"),temperature=0.1,model='llama-3.1-8b-instant',max_tokens=1024))
      def retrieve(self,query: str, top_k:int=5,score_threshold:float=0.0)-> List[Dict[str,Any]]:
          """ Retrieve relevant documents for a query 
          Args: 
          query: search query
          top_k: number of results to return
          score_threshold: Minimum similarity score threshold
          
          Returns:
          List of dictionaries containing retreived documents and metadata

          """
          print(f"Retreiving documnets for query: {query}")
          print(f"top_k: {top_k}, score_thrshold= {score_threshold}")

          #Generate query embedding 
          query_embedding=self.embedding_manager.generate_embeddings([query])[0]

          # Search in vector store
          try:
              results=self.vector_store.collection.query(
                  
                  query_embeddings=[query_embedding.tolist()],
                  n_results=top_k
              ) 
              # process results
              retrieved_docs=[]

              if results['documents'] and results['documents'][0]:
                  documents= results['documents'][0]
                  metadatas=results['metadatas'][0]
                  distances=results['distances'][0]
                  ids=results['ids'][0]

                  for i,(doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                      # Convert distance to similarity score (Chroma DB uses cosine similarity)
                      similarity_score= 1-distance

                      if similarity_score>=score_threshold:
                          retrieved_docs.append(
                              {
                                  'id':doc_id,
                                  'content':document,
                                  'metadata': metadata,
                                  'distance':distance,
                                  'similarity_score':similarity_score,
                                  'rank': i+1
                              }
                          )
                
                  print(f"Retrieved{len(retrieved_docs)} docs after filtering")
              else:
                  print("No Documents found for retreiving based on similarity search")
              return retrieved_docs    
          except Exception as e:
              print(f"Error found during retireval: {e}")
              return[]


      def Advanced_RAG(self,query:str, retriever,llm, top_k=5,min_score=0.2):
            
            """ RAG Pipeline with extra features
            returns answers,sources, confidence score, and optionaly full context
            """
            results=retriever.retrieve(query,top_k,score_threshold=min_score)
            
            if not results:
                return {"Answer":"No relevant context found", "sources":[],
                        "Confidence":0.0, "Context":" "}
            
            # Prepare Context and sources
            context="\n\n".join([doc['content']for doc in results]if results else"")
            sources=[{
                    'source':doc['metadata'].get('source_file',doc['metadata'].get('source','unknown')),
                    'page':doc['metadata'].get('page','unknown'),
                    'score':doc['similarity_score'],
                    'preview':doc['content'][:500]+'......',} for doc in results]
            confidence= max([doc['similarity_score'] for doc in results])

            # Generate Anser using GROQ LLM

            prompt=f""" 
                    use the following context and answer the query concisely.
                    
                    Context: {context},
                    Question: {query},
                    Answer:"""
            
            response=llm.invoke([prompt])
            output={

            'Answer': response.content,
            'Sources': sources,
            'Confidence_Score': confidence,
            'Context':context
            }

            return output
