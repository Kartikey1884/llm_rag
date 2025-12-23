import os
import time
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================================
# 1. App Configuration
# ======================================================
st.set_page_config(page_title="Mini RAG QA", layout="wide")
st.title("📄 Mini RAG – Document Question Answering")

st.markdown(
    """
Upload a document and ask questions.  
Answers are generated **strictly from the uploaded document** using Retrieval-Augmented Generation (RAG).
"""
)

# ======================================================
# 2. Sidebar (API Key + Config)
# ======================================================
st.sidebar.header("Configuration")

# 🔐 User-provided GROQ API key
user_api_key = st.sidebar.text_input(
    "Enter your GROQ API Key",
    type="password",
    help="Your API key is used only for this session and is never stored."
)

TOP_K = st.sidebar.slider("Top-K Retrieved Chunks", 1, 5, 3)

with st.sidebar.expander("ℹ️ Get GROQ API Key"):
    st.markdown("[Create a free GROQ API key](https://console.groq.com/keys)")

# ======================================================
# 3. Embeddings
# ======================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================================================
# 4. LLM (Initialized only after API key is provided)
# ======================================================
llm = None

if user_api_key:
    os.environ["GROQ_API_KEY"] = user_api_key

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0
    )
else:
    st.sidebar.info("🔑 Please enter your GROQ API key to enable QA.")

# ======================================================
# 5. Strict RAG Prompt (No Hallucination)
# ======================================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a question-answering assistant.

You MUST answer the question using ONLY the provided context.
If the answer is not explicitly stated, respond with:
"I don't know based on the provided documents."

Do NOT use prior knowledge.
Do NOT make assumptions.
"""
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ]
)

chain = prompt | llm | StrOutputParser() if llm else None

# ======================================================
# 6. Session State
# ======================================================
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ======================================================
# 7. Document Upload & Ingestion
# ======================================================
uploaded_file = st.file_uploader(
    "Upload a document (PDF or TXT)",
    type=["pdf", "txt"]
)

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Select loader
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    else:
        loader = TextLoader(temp_path)

    documents = loader.load()

    # --------------------------------------------------
    # Chunking Strategy:
    # - Chunk size: 400 tokens
    # - Overlap: 100 tokens
    # Balances semantic completeness with retrieval accuracy
    # --------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.session_state.retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    os.remove(temp_path)
    st.success("✅ Document indexed successfully")

# ======================================================
# 8. Query Interface
# ======================================================
st.subheader("Ask a Question")

st.caption("Sample test questions:")
st.code(
    """
Give me the correct coded classification for the following diagnosis:
"Recurrent depressive disorder, currently in remission"

What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?
"""
)

question = st.text_input("Enter your question")

# ======================================================
# 9. Query Execution
# ======================================================
if question:
    if llm is None:
        st.warning("⚠️ Please enter your GROQ API key in the sidebar.")
    elif st.session_state.retriever is None:
        st.warning("⚠️ Please upload a document first.")
    else:
        start = time.perf_counter()

        docs = st.session_state.retriever.invoke(question)

        if not docs:
            st.success("I don't know based on the provided documents.")
            st.stop()

        context = "\n\n".join(doc.page_content for doc in docs)

        answer = chain.invoke(
            {"context": context, "question": question}
        )

        elapsed = time.perf_counter() - start

        # ======================================================
        # 10. Output
        # ======================================================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔎 Retrieved Context")
            for i, doc in enumerate(docs, 1):
                st.markdown(
                    f"**Chunk {i} (Page {doc.metadata.get('page', 'N/A')}):**"
                )
                st.write(doc.page_content)

        with col2:
            st.subheader("✅ Final Answer")
            st.success(answer)
            st.caption(f"Response time: {elapsed:.2f}s")
