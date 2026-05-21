import streamlit as st
import requests
import time
import jieba
import re
import os
from openai import OpenAI
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

# ================= 1. 核心配置区 =================
SILICON_API_KEY = os.getenv("SILICON_API_KEY", "your-api-key-here")
SILICON_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

client = OpenAI(api_key=SILICON_API_KEY, base_url=SILICON_BASE_URL)
milvus_client = MilvusClient("rag_milvus_demo.db")

# ================= 2. 全局状态初始化 =================
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "bm25_model" not in st.session_state:
    st.session_state.bm25_model = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 3. 基础工具函数 =================

def get_embedding(text):
    response = client.embeddings.create(input=text, model=EMBED_MODEL)
    return response.data[0].embedding

def get_rerank_scores(query, docs):
    if not docs:
        return []
    url = f"{SILICON_BASE_URL}/rerank"
    payload = {"model": RERANK_MODEL, "query": query, "texts": docs, "return_documents": False}
    headers = {"Authorization": f"Bearer {SILICON_API_KEY}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            results = response.json().get("results", [])
            results.sort(key=lambda x: x["index"])
            return [r["relevance_score"] for r in results]
    except Exception as e:
        st.error(f"重排序失败: {e}")
    return [0] * len(docs)

def tokenize(text):
    return list(jieba.cut_for_search(text))

def smart_chunk_text(text, chunk_size, chunk_overlap):
    """【升级】感知段落的切块算法，尽量不切断制度条款"""
    paragraphs = re.split(r'\n+', text)
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(p) >= chunk_size: # 如果单段超长，再用滑窗
                start = 0
                while start < len(p):
                    chunks.append(p[start:start+chunk_size])
                    start += (chunk_size - chunk_overlap)
                current_chunk = ""
            else:
                current_chunk = p + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ================= 4. 数据库与索引构建 =================

def init_db_and_index(chunks):
    new_collection = f"rag_docs_{int(time.time())}"
    milvus_client.create_collection(
        collection_name=new_collection, dimension=1024, metric_type="COSINE"
    )

    data = []
    progress_bar = st.progress(0, text="正在计算向量并存入 Milvus...")
    for i, chunk in enumerate(chunks):
        vec = get_embedding(chunk)
        data.append({"id": i, "vector": vec, "text": chunk})
        progress_bar.progress((i + 1) / len(chunks))

    milvus_client.insert(collection_name=new_collection, data=data)
    milvus_client.load_collection(new_collection)

    tokenized_corpus = [tokenize(doc) for doc in chunks]
    st.session_state.bm25_model = BM25Okapi(tokenized_corpus)
    st.session_state.collection_name = new_collection
    st.session_state.doc_chunks = chunks

    progress_bar.empty()
    st.success(f"✅ 处理成功！共 {len(chunks)} 个分块已存入双路检索引擎。")

# ================= 5. 检索策略实现 =================

def naive_rag_search(query):
    if not st.session_state.collection_name: return []
    query_vec = get_embedding(query)
    search_res = milvus_client.search(
        collection_name=st.session_state.collection_name, data=[query_vec], limit=3, output_fields=["text"]
    )
    return [hit["entity"]["text"] for hit in search_res[0]]

def rrf_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    for rank, doc in enumerate(dense_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

def advanced_rag_pipeline(query, status):
    if not st.session_state.collection_name: return [], {}
    debug_log = {}

    # 【升级 1】纯净拆解（抛弃容易引发幻觉的 HyDE）
    status.update(label="🧠 [查询优化] 正在拆解复杂问题...", state="running")
    opt_prompt = f"""请分析以下用户问题。如果是包含多个条件的复杂问题，请拆分为简单子问题。
只输出拆分后的子问题文本，每行一个，不要输出编号和多余废话。如果问题很简单，直接原样输出。
用户问题：{query}"""
    
    sub_queries_text = client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role": "user", "content": opt_prompt}], max_tokens=150
    ).choices[0].message.content.strip()
    
    sub_queries = [sq.strip() for sq in sub_queries_text.split('\n') if sq.strip()]
    if not sub_queries: sub_queries = [query]
    debug_log["sub_queries"] = sub_queries

    final_all_docs = []
    
    # 【升级 2】独立检索引擎：为每个子问题单独检索、单独重排、单独保底
    status.update(label="🔍 [检索优化] 为各个子问题独立执行双路检索与重排...", state="running")
    for sq in sub_queries:
        # 稠密检索
        vec = get_embedding(sq)
        res_m = milvus_client.search(
            collection_name=st.session_state.collection_name, data=[vec], limit=5, output_fields=["text"]
        )
        dense_docs = [hit["entity"]["text"] for hit in res_m[0]]
        
        # 稀疏检索 (BM25)
        tokenized_q = tokenize(sq)
        bm25_scores = st.session_state.bm25_model.get_scores(tokenized_q)
        top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
        sparse_docs = [st.session_state.doc_chunks[i] for i in top_n_idx]
        
        # RRF 融合
        unique_dense = list(dict.fromkeys(dense_docs))
        unique_sparse = list(dict.fromkeys(sparse_docs))
        fused_docs = rrf_fusion(unique_dense, unique_sparse)[:10]
        
        # 独立重排与压缩
        scores = get_rerank_scores(sq, fused_docs)
        scored_docs = sorted(zip(fused_docs, scores), key=lambda x: x[1], reverse=True)
        
        # 每个子问题强制保留最相关的 Top-2 证据（防止被其他子问题的长文本挤掉）
        retained_docs = [doc for doc, score in scored_docs[:2]]
        final_all_docs.extend(retained_docs)

    # 去除跨子问题重复的文本块
    unique_final_docs = list(dict.fromkeys(final_all_docs))
    debug_log["final_compressed_count"] = len(unique_final_docs)

    return unique_final_docs, debug_log

# ================= 6. LLM 调用封装 =================

def llm_chat(system_prompt, user_prompt):
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=20
    )
    return res.choices[0].message.content.strip()

def llm_stream(system_prompt, user_prompt, placeholder):
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        stream=True
    )
    full_response = ""
    crag_triggered = False
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            full_response += delta
            if "[CRAG_FALLBACK]" in full_response:
                crag_triggered = True
                break
            placeholder.markdown(full_response + "▌")

    if crag_triggered:
        full_response = "🌐 **[CRAG 机制触发]**：对不起，检索到的文档中未能找到回答该问题所需的明确规定或数据。\n\n*系统已拦截幻觉，避免提供不准确的推测信息。*"

    placeholder.markdown(full_response)
    return full_response

# ================= 7. UI 主界面 =================

st.set_page_config(page_title="优化 RAG 架构演示", layout="wide")
st.title("🚀 工业级优化 RAG vs 基础 RAG (制度问答优化版)")

with st.sidebar:
    st.header("📁 第一步：构建知识库")
    uploaded_file = st.file_uploader("上传 TXT 格式长文档", type=["txt"])

    st.subheader("⚙️ 动态分块参数")
    chunk_size = st.slider("Chunk Size（建议调大以保护制度条款）", 300, 2000, 800, step=50)
    chunk_overlap = st.slider("Chunk Overlap（重叠区间）", 0, 500, 200, step=25)

    if st.button("🔨 构建混合检索引擎", type="primary"):
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            chunks = smart_chunk_text(text, chunk_size, chunk_overlap)
            init_db_and_index(chunks)
        else:
            st.error("请先上传 TXT 文件！")

    st.markdown("---")
    st.header("🔀 第二步：切换 RAG 模式")
    rag_mode = st.radio("选择体验模式：", ["🗑️ 基础 RAG", "💎 优化 RAG (制度特化版)"], index=1)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("向知识库提问（如：主管岗转正和请假超过3天分别需要谁审批？）..."):
    if not st.session_state.collection_name:
        st.warning("⚠️ 请先在侧边栏上传文档并构建知识库！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = "你是一个友好的 AI 助手，请直接回答用户的问题。"
    context_str = ""
    skip_retrieval = False

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        if rag_mode.startswith("💎"):
            with st.spinner("⚡ [Self-RAG] 正在判断是否需要检索..."):
                route_result = llm_chat(
                    system_prompt="意图分类器：若用户输入为打招呼或极简常识，回复NO；若涉及制度、数据、文档提问，回复YES。",
                    user_prompt=prompt
                )
            if "NO" in route_result.upper():
                skip_retrieval = True
                st.info("⚡ **[Self-RAG]** 判定为日常对话，跳过检索。")
            else:
                skip_retrieval = False

        if not skip_retrieval:
            if rag_mode.startswith("🗑️"):
                with st.status("🔍 执行基础向量检索...", expanded=True) as status:
                    context_docs = naive_rag_search(prompt)
                    context_str = "\n\n---\n\n".join(context_docs)
                    status.update(label="✅ 基础检索完成", state="complete", expanded=False)

                system_prompt = f"请基于以下文档内容回答用户问题。\n\n【文档内容】：\n{context_str}"
            else:
                with st.status("🚀 触发优化 RAG (制度问答优化版)...", expanded=True) as status:
                    context_docs, debug_log = advanced_rag_pipeline(prompt, status)
                    context_str = "\n\n---\n\n".join(context_docs)
                    status.update(label="✅ 全链路检索完成！", state="complete", expanded=False)

                with st.expander("🔬 查看检索过程详情（独立拆解与保底机制）"):
                    st.json(debug_log)

                # 【升级 3】严厉的反幻觉与反推测 Prompt
                system_prompt = f"""你是一个严谨的企业合规与财务问答助手。请严格遵守以下红线指令：

【严格红线指令】：
1. 你必须完全基于下方【检索到的文档上下文】作答，提取相关的流程、审批人或数据。
2. 绝对禁止脑补：若文档中未明确说明某项规定（例如没有明确写明审批层级），必须回答“文档中未明确说明”，绝不允许使用“推测”、“类似于”、“逻辑上”等词汇进行类比或自由发挥。
3. 如果所有文档内容与问题完全无关，请在回答开头输出特殊标记：[CRAG_FALLBACK]，然后停止。
4. 在回答不同子问题时，请分点清晰列出，条理分明。

【检索到的文档上下文】：
{context_str}"""

        full_response = llm_stream(system_prompt, prompt, message_placeholder)

    st.session_state.messages.append({"role": "assistant", "content": full_response})