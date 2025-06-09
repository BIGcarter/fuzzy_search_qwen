"""
arxiv_semantic_search.py
示例依赖:
    pip install "transformers>=4.51.0" faiss-cpu torch datasets tqdm
"""

import json, os, faiss, torch, torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# --------------------------- 1. 模型与超参 ---------------------------
EMB_MODEL_NAME   = "Qwen/Qwen3-Embedding-4B"
RERANK_MODEL     = "Qwen/Qwen3-Reranker-4B"   # 若不用精排，可不加载
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN          = 2048                         # 根据显存可调
TOP_K_RECALL     = 100                          # ANN 召回数
TOP_K_FINAL      = 10                           # 最终展示给用户的数目
INDEX_FILE       = "arxiv_demo.index"
DOC_STORE_FILE   = "arxiv_docs.jsonl"           # 存文档元信息，方便展示

# --------------------------- 2. 工具函数 ---------------------------
def last_token_pool(last_hidden, attn_mask):
    """官方推荐的 embedding pooling：取每条序列最后一个非 padding token"""
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
    instr = "Given a web search query, retrieve relevant passages that answer the query"
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
        pairs.append(f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
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
    print("▶ 下载/读取示例数据（100 篇论文）")
    ds = load_dataset("ccdv/arxiv-classification", split="train[:100]")
    # 文档格式：Title + Abstract
    docs_jsonl = []
    texts = []
    for row in ds:
        text = f"Title: {row['title']}\nAbstract: {row['abstract']}"
        texts.append(text)
        docs_jsonl.append({"id": len(docs_jsonl), "title": row["title"], "text": text})

    print("▶ 加载 Qwen3-Embedding 模型")
    tok_emb = AutoTokenizer.from_pretrained(EMB_MODEL_NAME, padding_side="left")
    model_emb = AutoModel.from_pretrained(EMB_MODEL_NAME).to(DEVICE).eval()

    print("▶ 计算文档 embedding")
    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_vec = embed_texts(texts[i:i+batch_size], tok_emb, model_emb)
        embeddings.append(batch_vec.cpu())
    embeddings = torch.cat(embeddings).numpy().astype("float32")

    print("▶ 构建 Faiss 向量索引")
    index = faiss.IndexFlatIP(embeddings.shape[1])       # 内积 = 归一化后 cosine
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(DOC_STORE_FILE, "w") as f:
        for obj in docs_jsonl:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ 索引完成：{len(docs_jsonl)} 篇文档, index => {INDEX_FILE}")

# --------------------------- 4. 在线阶段：查询 ---------------------------
def online_search():
    if not Path(INDEX_FILE).exists():
        print("❗ 请先运行 offline_build() 建库")
        return

    print("▶ 加载索引 & 文档元信息")
    index = faiss.read_index(INDEX_FILE)
    docs  = [json.loads(line) for line in open(DOC_STORE_FILE, encoding="utf-8")]
    tok_emb = AutoTokenizer.from_pretrained(EMB_MODEL_NAME, padding_side="left")
    model_emb = AutoModel.from_pretrained(EMB_MODEL_NAME).to(DEVICE).eval()

    # ⚠️ 若要启用精排：
    use_rerank = True
    if use_rerank:
        tok_rr  = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        model_rr = AutoModelForCausalLM.from_pretrained(RERANK_MODEL).to(DEVICE).eval()

    while True:
        raw_query = input("\n❓ 输入检索语句(回车退出)：").strip()
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

        print("\n🔎 Top 结果：")
        for rank, (doc_id, score) in enumerate(reranked, 1):
            paper = docs[doc_id]
            title = paper["title"][:120].replace("\n", " ")
            print(f"{rank:02d}. [{score:.3f}] {title}")

# --------------------------- 5. 入口 ---------------------------
if __name__ == "__main__":
    if not Path(INDEX_FILE).exists():
        offline_build()   # 第一次运行会自动建库（约 2~3 分钟/CPU）
    online_search()
