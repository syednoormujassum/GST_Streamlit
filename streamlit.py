from main import *
import streamlit as st
from src.chunking import Chunking_Embedding_Manager
from src.vectorstore import VectorStore
from src.search_retreiver import RAGRetriever,ChatGroq,load_dotenv,os
import pandas as pd
import json

embedding_manager = Chunking_Embedding_Manager()
vector_store = VectorStore()
rag_retirever=RAGRetriever(vector_store,embedding_manager)
rag_retirever=RAGRetriever(vector_store,embedding_manager)
llm=(ChatGroq(groq_api_key=os.getenv("GROQ_API"),temperature=0.1,model='llama-3.1-8b-instant',max_tokens=1024))
#     answer=rag_retirever.Advanced_RAG("describe about GST?",rag_retirever,llm=llm)

st.set_page_config(layout="wide",page_title="GST_ChatBot",page_icon="💬")
st.title("💬 GST Bot")
st.sidebar.image("./dgh_icon.png")
st.sidebar.text("Founder: Syed Noor Mujassum.")

st.divider()

with st.container(border=True):
    query=st.text_input("Enter Your Query")
    answer=rag_retirever.Advanced_RAG(query,rag_retirever,llm=llm)
    if st.button("Ask"):
        # df=pd.DataFrame(answer)
        # st.write(df[['Answer']])
        
        with st.chat_message("assistant",width="stretch"):
            st.json(answer)
