import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

st.title("InsightHub: News Analysis & Q&A")

# --- URL Input ---
url = st.text_input("Enter a news article URL:")
main_placeholder = st.empty()

if url:
    with st.spinner("Loading content..."):
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embed
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Use Gemini LLM from LangChain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # or gemini-1.5-pro

        # QA Chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Question input
        query = st.text_input("Ask a question based on this webpage:")

        if query:
            docs_with_query = vectorstore.similarity_search(query)
            answer = chain.run(input_documents=docs_with_query, question=query)
            st.markdown("### 💬 Answer:")
            st.write(answer)
