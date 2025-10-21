import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap  # ✅ correct import

# Get GROQ API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def create_qa_system(chunks):
    """
    Input: Chunks from scraper
    Output: Runnable QA system
    """
    # Step 1: Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    # Step 2: Initialize Groq LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")



    # Step 3: Define prompt
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Step 4: Build runnable chain (LCEL)
    qa_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
    )

    return qa_chain
