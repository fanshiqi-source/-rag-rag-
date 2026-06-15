import os
import streamlit as st

if "SILICON_API_KEY" not in os.environ:
    api_key = st.secrets.get("SILICON_API_KEY")
    if api_key:
        os.environ["SILICON_API_KEY"] = api_key

from pymilvus import MilvusClient
from nbrag.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from nbrag.llm_client import LLMClient
from nbrag.chunking import smart_chunk_text
from nbrag.indexing import build_index
from nbrag.retrieval import naive_rag_search
from nbrag.pipeline import advanced_rag_pipeline

llm = LLMClient()
milvus_client = MilvusClient("rag_milvus_demo.db")

for key, default in [
    ("collection_name", None),
    ("bm25_model", None),
    ("doc_chunks", []),
    ("messages", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(page_title="RAG 架构对比演示", layout="wide")
st.title("🚀 混合检索 RAG：基础版 vs 优化版")

with st.sidebar:
    st.header("📁 第一步：构建知识库")
    uploaded_file = st.file_uploader("上传 TXT 格式长文档", type=["txt"])
    st.subheader("⚙️ 动态分块参数")
    chunk_size = st.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=25)

    if st.button("🔨 构建混合检索引擎", type="primary"):
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            chunks = smart_chunk_text(text, chunk_size, chunk_overlap)
            with st.spinner("正在构建索引..."):
                col_name, bm25, docs = build_index(chunks, llm.embed, milvus_client)
                st.session_state.collection_name = col_name
                st.session_state.bm25_model = bm25
                st.session_state.doc_chunks = docs
            st.success(f"✅ 处理成功！共 {len(chunks)} 个分块已存入双路检索引擎。")
        else:
            st.error("请先上传 TXT 文件！")

    st.markdown("---")
    st.header("🔀 第二步：切换 RAG 模式")
    rag_mode = st.radio("选择检索模式：", ["🗑️ 基础 RAG", "💎 优化 RAG"], index=1)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("向知识库提问..."):
    if not st.session_state.collection_name:
        st.warning("⚠️ 请先构建知识库！")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = "你是一个友好的 AI 助手，请直接回答用户的问题。"
    context_str = ""
    skip_retrieval = False

    with st.chat_message("assistant"):
        placeholder = st.empty()

        if rag_mode.startswith("💎"):
            with st.spinner("⚡ [Self-RAG] 正在判断是否需要检索..."):
                route = llm.chat(
                    "意图分类器：若用户输入为打招呼或极简常识，回复NO；若涉及制度、数据、文档提问，回复YES。",
                    prompt
                )
            if "NO" in route.upper():
                skip_retrieval = True
                st.info("⚡ **[Self-RAG]** 判定为日常对话，跳过检索。")

        if not skip_retrieval:
            if rag_mode.startswith("🗑️"):
                with st.status("🔍 执行基础向量检索...", expanded=True) as status:
                    docs = naive_rag_search(
                        prompt,
                        st.session_state.collection_name,
                        milvus_client,
                        llm.embed
                    )
                    context_str = "\n\n---\n\n".join(docs)
                    status.update(label="✅ 基础检索完成", state="complete", expanded=False)
                system_prompt = f"请基于以下文档内容回答用户问题。\n\n【文档内容】：\n{context_str}"
            else:
                with st.status("🚀 触发优化 RAG...", expanded=True) as status:
                    docs, debug = advanced_rag_pipeline(
                        prompt,
                        st.session_state.collection_name,
                        milvus_client,
                        llm,
                        st.session_state.bm25_model,
                        st.session_state.doc_chunks
                    )
                    context_str = "\n\n---\n\n".join(docs)
                    status.update(label="✅ 全链路检索完成！", state="complete", expanded=False)
                with st.expander("🔬 查看检索过程详情"):
                    st.json(debug)
                system_prompt = f"""你是一个严谨的企业合规与财务问答助手。请严格遵守以下红线指令：

【严格红线指令】：
1. 你必须完全基于下方【检索到的文档上下文】作答，提取相关的流程、审批人或数据。
2. 绝对禁止脑补：若文档中未明确说明某项规定（例如没有明确写明审批层级），必须回答"文档中未明确说明"，绝不允许使用"推测"、"类似于"、"逻辑上"等词汇进行类比或自由发挥。
3. 如果所有文档内容与问题完全无关，请在回答开头输出特殊标记：[CRAG_FALLBACK]，然后停止。
4. 在回答不同子问题时，请分点清晰列出，条理分明。

【检索到的文档上下文】：
{context_str}"""

        full_response = ""
        crag_detected = False
        for delta in llm.chat_stream(system_prompt, prompt):
            full_response += delta
            placeholder.markdown(full_response + "▌")
            if "[CRAG_FALLBACK]" in full_response:
                crag_detected = True
                break

        if crag_detected:
            full_response = "🌐 **[CRAG 机制触发]**：对不起，检索到的文档中未能找到回答该问题所需的明确规定或数据。\n\n*系统已拦截幻觉，避免提供不准确的推测信息。*"

        placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
