# Assistant PDF Intelligent RAG 

A small Streamlit app that lets you upload PDF files and ask questions using a Retrieval-Augmented Generation (RAG) workflow powered by LangChain, FAISS, and OpenAI models.

## Features

- Upload one or more documents from the sidebar
- Build an in-memory FAISS vector index from document chunks
- Ask natural-language questions in a chat interface
- Retrieve relevant context before generating answers

## Requirements

- Python 3.12+
- An OpenAI API key

## Quick Start

```bash
# from project root
pip install -e .
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal, upload a PDF, and start asking questions.

## Project Structure

```text
app.py                # Streamlit UI entrypoint
src/ingest.py         # Document loading, chunking, and FAISS index creation
src/rag_agent.py      # Agent construction with retrieval tool
src/tool.py           # Similarity search tool wrapper
src/models.py         # LLM and embedding model setup
```
