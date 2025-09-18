import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from pypdf import PdfReader
import textwrap, os, glob
import chromadb
from chromadb.utils import embedding_functions

def clean_text(text):
    # Remove LaTeX math notation
    text = re.sub(r'\\\(.*?\\\)', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# INGEST
def read_pdf(path):
    txt = []
    reader = PdfReader(path)
    
    for p in reader.pages:
        t = p.extract_text() or ""
        txt.append(t)
    return "\n".join(txt)

def chunk(text, size=800, overlap=120):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return [c.strip() for c in out if c.strip()]

def load_corpus(folder='docs'):
    docs = []
    for f in glob.glob(os.path.join(folder, "*.pdf")):
        text = read_pdf(f)
        docs.append((os.path.basename(f), clean_text(text)))
    return docs

# GENERATE WITH QWEN
def ask_qwen(question, context, tok, gen, max_new_tokens=350, temperature=0.3):
    sys = "You are a helpful assistant. Answer using only the given context. If missing, say what is missing. if not found, say there is no infomation."
    user = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and cite snippets."
    messages = [{"role":"system","content":sys},{"role":"user","content":user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(gen.device)
    out = gen.generate(**inputs, 
                       max_new_tokens=max_new_tokens, 
                       do_sample=True,
                       temperature=temperature, 
                       top_p=0.9)

    text = tok.decode(out[0], skip_special_tokens=True)

    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple RAG agent')
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Number of top contents to retrieve (default: 10)")
    parser.add_argument("-i", "--ingest", action="store_true", help="Ingest local docs")

    args = parser.parse_args()  

    GEN_ID = "Qwen/Qwen2-1.5B-Instruct"
    EMB_ID = "sentence-transformers/all-MiniLM-L6-v2"
    docs_dir = "docs"

    # Generator 
    tok = AutoTokenizer.from_pretrained(GEN_ID)
    gen = AutoModelForCausalLM.from_pretrained(GEN_ID, torch_dtype='auto', device_map='auto')

    # Chroma
    client = chromadb.PersistentClient(path="chroma_db")
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_ID)
    coll = client.get_or_create_collection(
        name="rag_docs",
        embedding_function=emb_func,
        metadata={"hnsw:space": "cosine"}
    )

    # Ingest -> chunk -> upsert
    if args.ingest:
        raw_docs = load_corpus()
        ids, documents, metadatas = [], [], []
        for name, txt in raw_docs:
            i = 0
            for c in chunk(txt):  # your existing chunk() function
                ids.append(f"{name}-{i}")
                documents.append(f"[{name}] {c}")
                metadatas.append({"source": name, "chunk": i})
                i += 1
       
        try:
            coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except Exception:
            coll.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Ingested {len(documents)} chunks into Chroma at ./chroma_db")

    try:
        while (1):
            q = input("Enter your question: ")
            if not q:
                continue

            res = coll.query(
                query_texts=[q],
                n_results=int(args.top_k),
                include=["documents", "metadatas", "distances"]
            )

            top = res["documents"][0] if res and res.get("documents") else []
            context = "\n\n----\n\n".join(top)
            print(textwrap.fill(ask_qwen(q, context, tok, gen), 100))

    except KeyboardInterrupt:
        print("Stop running ...")