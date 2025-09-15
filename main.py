import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss, numpy as np, textwrap, os, glob

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

# BUILD INDEX
def build_index(emb, chunks):
    if not isinstance(chunks, list):
        chunks = list(chunks)
    chunks = [clean_text(c) for c in chunks]
    
    vecs = emb.encode(chunks, normalize_embeddings=True, batch_size=32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype(np.float32))
    
    return index, vecs

# 4) RETRIEVE
def retrieve(emb, query, chunks, index, k=4):
    qv = emb.encode([query], normalize_embeddings=True).astype(np.float32)
    D, I = index.search(qv, k)
    return [chunks[i] for i in I[0]]

# 5) GENERATE WITH QWEN
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
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=10,
        help="Number of top contents to retrieve (default: 10)"
    )

    args = parser.parse_args()  

    GEN_ID = "Qwen/Qwen2-1.5B-Instruct"
    EMB_ID = "sentence-transformers/all-MiniLM-L6-v2"
    docs_dir = "docs"

    tok = AutoTokenizer.from_pretrained(GEN_ID)
    gen = AutoModelForCausalLM.from_pretrained(GEN_ID, torch_dtype='auto', device_map='auto')
    emb = SentenceTransformer(EMB_ID)

    raw_docs = load_corpus(docs_dir)

    all_chunks = []
    for name, txt in raw_docs:
        for c in chunk(txt):
            all_chunks.append(f"[{name}] {c}")

    index, vecs = build_index(emb, all_chunks)

    try:
        while (1):
            q = input("Enter your question: ")
            top = retrieve(emb, q, all_chunks, index, k=int(args.top_k))
            context = "\n\n----\n\n".join(top)
            print(textwrap.fill(ask_qwen(q, context, tok, gen), 100))
    except KeyboardInterrupt:
        print("Stop running ...")