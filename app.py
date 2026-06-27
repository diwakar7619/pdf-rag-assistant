import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
st.title("📄 RAG Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is None:
    st.info("Please upload a PDF to begin.")
    st.stop()


with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
    temp_file.write(uploaded_file.getvalue())
    temp_path = temp_file.name


@st.cache_resource
def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
)

vector_store = create_vector_store(temp_path)

question = st.text_input("Ask a question about your PDF")

if question:
    results = vector_store.similarity_search(question)

    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
Answer the user's question using ONLY the context below.

If the answer is not present in the context, say:
"I couldn't find that information in the document."

Context:
{context}

Question:
{question}
"""

    with st.spinner("Searching document..."):
        response = llm.invoke(prompt)

    st.markdown("### Answer")
    st.write(response.content)
