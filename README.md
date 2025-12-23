# Mini LLM-Powered Question Answering System (RAG)

## Overview
This project implements a Mini Retrieval-Augmented Generation (RAG) based Question-Answering system developed as part of a Software Engineering Intern technical assessment.

The system allows users to upload a document (PDF or TXT) and ask questions. Answers are generated strictly from the uploaded document to avoid hallucinations.

The solution focuses on delivering a clean, functional MVP within the given time constraints.

---

## Objective
- Build an end-to-end RAG pipeline
- Perform document ingestion, chunking, embedding, retrieval, and generation
- Use open-source or free-tier accessible LLMs
- Demonstrate sound engineering decisions under time pressure

---

## System Architecture

Document Upload  
→ Document Loader  
→ Text Chunking  
→ Embedding Generation  
→ Vector Store (FAISS)  
→ Top-K Retrieval  
→ Strict Prompt Grounding  
→ LLM Answer Generation  

---

## Tools & Technologies Used

- Language: Python  
- UI: Streamlit  
- Document Loaders: PyPDFLoader, TextLoader  
- Chunking: RecursiveCharacterTextSplitter  
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)  
- Vector Store: FAISS  
- LLM: LLaMA-3.1-8B (via Groq)  
- Framework: LangChain  

---

## Document Ingestion & Chunking Strategy

- Supported formats: PDF, TXT  
- Chunk size: 400 tokens  
- Chunk overlap: 100 tokens  

This strategy balances semantic completeness with efficient retrieval and prevents loss of context across chunk boundaries.

---

## Embedding & Retrieval

- Embeddings are generated using a pre-trained MiniLM model
- FAISS is used for fast in-memory similarity search
- Top-K relevant chunks are retrieved for each query

---

## LLM Integration

- Uses LLaMA-3.1-8B via Groq (free-tier accessible)
- Strict prompt ensures answers are generated only from retrieved context
- If information is missing, the model responds with:
  "I don't know based on the provided documents."

---

## Sample Queries

- Give me the correct coded classification for the following diagnosis:
  "Recurrent depressive disorder, currently in remission"

- What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?

---

## API Key Handling

- Users provide their own GROQ API key via the sidebar
- Keys are never stored or logged
- Keys exist only for the current session

---

## AI Tool Usage

ChatGPT was used for:
- Prompt design refinement
- Code structure validation
- Documentation drafting

All final decisions were reviewed and customized.

---

## Limitations

- No reranking (e.g., MMR)
- No persistent vector storage
- Optimized for structured text documents
- Minimal error handling due to time constraints

---

## How to Run

pip install -r requirements.txt  
streamlit run app.py  

---

## Conclusion

This project demonstrates a clean, grounded, and functional RAG-based QA system built under a strict time limit with a focus on clarity, correctness, and simplicity.
