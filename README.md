# Simple RAG Agent (Chroma + Qwen2-1.5B-Instruct)

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline in Python.  
It ingests PDFs, chunks & embeds them, **stores vectors locally with ChromaDB**, and answers questions using **Qwen/Qwen2-1.5B-Instruct** with context and citations.

---

## 🚀 Features
- 📄 **PDF ingestion** via `pypdf.PdfReader` (no OCR required)
- ✂️ **Chunking** with overlap for longer documents
- 🧠 **Local, persistent vector store** using **ChromaDB**
- 🔍 **Semantic search** with **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- 🤖 **Generation** by **Qwen2-1.5B-Instruct** (Transformers)
- ⌨️ **Interactive CLI**: type questions and get cited answers

---

## 🛠️ Tech Stack
- **Generator**: [Transformers](https://github.com/huggingface/transformers) — `Qwen/Qwen2-1.5B-Instruct`
- **Embeddings**: [Sentence-Transformers](https://www.sbert.net/) — `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store**: [ChromaDB](https://www.trychroma.com/) (local persistence under `./chroma_db`)
- **PDF parsing**: [PyPDF](https://pypi.org/project/pypdf/)
- **Runtime**: [PyTorch](https://pytorch.org/)

---

## 📦 Installation

```bash
# PyTorch (choose wheels appropriate for your OS/GPU)
pip install torch torchvision torchaudio

# Core libraries
pip install transformers sentence-transformers pypdf chromadb
```

> CPU-only is fine for Qwen 1.5B (slower). A small GPU will speed up inference.

---

## 📁 Project Layout

```
rag_agent.py            # your script (this repo)
docs/                   # put your PDFs here
chroma_db/              # auto-created Chroma database (persistent)
```

---

## ⚙️ Defaults (as in the script)

- **Generator**: `GEN_ID = "Qwen/Qwen2-1.5B-Instruct"`  
- **Embeddings**: `EMB_ID = "sentence-transformers/all-MiniLM-L6-v2"`  
- **Docs directory**: `docs/`  
- **Top-k retrieved chunks**: configurable via `-k / --top_k` (default `10`)  
- **Chroma path**: `./chroma_db`

---

## ▶️ Usage

### 1) Add PDFs
Place one or more PDFs into `./docs/`.  

### 2) Ingest (build/update the Chroma index)
```bash
python rag_agent.py --ingest
```
This will:
- extract text from each PDF,
- chunk with overlap,
- embed each chunk,
- **persist** vectors and metadata in `./chroma_db`.

### 3) Ask questions (retrieves from Chroma)
```bash
python rag_agent.py -k 5
```
Then type your question at the prompt. The app queries Chroma for the **top-k** chunks and sends them as context to Qwen for answer generation.

---
