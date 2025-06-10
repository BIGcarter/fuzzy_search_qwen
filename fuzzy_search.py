import json, os, faiss, torch, torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# --------------------------- 1. 模型与超参 ---------------------------
EMB_MODEL_NAME    = "Qwen/Qwen3-Embedding-4B"
RERANK_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"   # 若不用精排，可不加载
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN           = 2048                         # 根据显存可调
TOP_K_RECALL      = 50                          # ANN 召回数
TOP_K_FINAL       = 10                           # 最终展示给用户的数目
INDEX_FILE        = "arxiv_synthetic_demo.index"
DOC_STORE_FILE    = "synthetic_corpus.jsonl"           
CORPUS_FILE       = "synthetic_corpus.jsonl"        
MODEL_DIR         = "/Users/maixiaofeng/.cache/modelscope/hub/models/" 
# EMB_MODEL         = MODEL_DIR + EMB_MODEL_NAME
# RERANK_MODEL      = MODEL_DIR + RERANK_MODEL_NAME
# assert Path(EMB_MODEL).exists(), f"Embedding model not found: {EMB_MODEL}"
# assert Path(RERANK_MODEL).exists(), f"Rerank model not found: {RERANK_MODEL}"
EMB_MODEL         = "Qwen/Qwen3-Embedding-0.6B"   
RERANK_MODEL      = "None"

# --------------------------- 2. 工具函数 ---------------------------   
def last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor):
    seq_len = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.size(0)), seq_len]

def embed_texts(texts: List[str], tokenizer, model) -> torch.Tensor:
    tok = tokenizer(texts, truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**tok).last_hidden_state
    vec = last_token_pool(out, tok["attention_mask"])
    return F.normalize(vec, p=2, dim=1)

def build_instruction_query(raw_query: str) -> str:
    instr = "Given a paper search query of certain field, retrieve relevant passages that answer the query"
    return f"Instruct: {instr}\nQuery: {raw_query}"

def rerank(
    query: str,
    cand_indices: List[int],
    docs: List[dict],
    tokenizer_rr,
    model_rr
) -> List[Tuple[int, float]]:
    """对召回的文档做精排，返回 [(doc_id, score)]"""
    pairs = []
    for idx in cand_indices:
        doc = docs[idx]["text"]
        pairs.append(f"<Instruct>: Given a paper search query of certain field, retrieve relevant passages that answer the query\n"
                     f"<Query>: {query}\n"
                     f"<Document>: {doc}")
    # 批量推理
    tok = tokenizer_rr(pairs, padding=True, truncation=True,
                       max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model_rr(**tok).logits[:, -1, :]
    yes_id = tokenizer_rr.convert_tokens_to_ids("yes")
    no_id  = tokenizer_rr.convert_tokens_to_ids("no")
    scores = torch.softmax(logits[:, [no_id, yes_id]], dim=1)[:, 1]  # 取 "yes" 概率
    return sorted(zip(cand_indices, scores.tolist()),
                  key=lambda x: x[1], reverse=True)

# --------------------------- 3. 离线阶段：构建向量库 ---------------------------
def offline_build():
    ds = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                ds.append(data)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e} for line: {line.strip()}")
    texts = []
    for row in ds:
        text = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        texts.append(text)

    tok_emb = AutoTokenizer.from_pretrained(EMB_MODEL, padding_side="left")
    model_emb = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()

    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_vec = embed_texts(texts[i:i+batch_size], tok_emb, model_emb)
        embeddings.append(batch_vec.cpu())
    embeddings = torch.cat(embeddings).numpy().astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])      
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    print(f"索引完成：{len(ds)} 篇文档, index => {INDEX_FILE}")

# --------------------------- 4. 在线阶段：查询 ---------------------------
def online_search():
    if not Path(INDEX_FILE).exists():
        print("请先运行 offline_build() 建库")
        return

    index = faiss.read_index(INDEX_FILE)
    docs  = [json.loads(line) for line in open(DOC_STORE_FILE, encoding="utf-8")]
    tok_emb = AutoTokenizer.from_pretrained(EMB_MODEL, padding_side="left")
    model_emb = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()

    # 若要启用精排：
    use_rerank = False
    if use_rerank:
        tok_rr  = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        model_rr = AutoModelForCausalLM.from_pretrained(RERANK_MODEL).to(DEVICE).eval()

    while True:
        raw_query = input("\nEnter you topic (Press ENTER to exit)：").strip()
        if not raw_query:
            break

        q_text = build_instruction_query(raw_query)
        q_vec  = embed_texts([q_text], tok_emb, model_emb).cpu().numpy().astype("float32")

        D, I = index.search(q_vec, TOP_K_RECALL)   # I: indices  D: similarity
        cand_indices = I[0].tolist()

        if use_rerank:
            reranked = rerank(raw_query, cand_indices, docs, tok_rr, model_rr)[:TOP_K_FINAL]
        else:
            reranked = list(zip(cand_indices, D[0]))[:TOP_K_FINAL]

        print("\nTop 结果：")
        for rank, (doc_id, score) in enumerate(reranked, 1):
            paper = docs[doc_id]
            title = paper["title"][:120].replace("\n", " ")
            print(f"{rank:02d}. [{score:.3f}] {title}")

# --------------------------- 5. 入口 ---------------------------
if __name__ == "__main__":
    if not Path(INDEX_FILE).exists():
        offline_build()   
    online_search()
