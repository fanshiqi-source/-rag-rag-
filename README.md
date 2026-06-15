# NBRAG – 混合检索 RAG 系统

基于 Streamlit 的双模式 RAG（检索增强生成）系统，支持基础 RAG 与优化 RAG 对比，集成稠密向量检索、BM25 稀疏检索与重排序的混合检索流水线。

## 功能特性

- **双模式对比**：基础 RAG vs 优化 RAG，支持检索质量与生成结果直观对比
- **查询分解**：自动将多条件复杂问题拆解为独立子问题分别检索
- **混合检索**：稠密向量检索 + BM25 稀疏检索 + RRF 融合 + 重排序
- **事实一致性保障**：Self-RAG 路由 + CRAG 回退机制，抑制无依据生成
- **模块化设计**：config / llm_client / chunking / indexing / retrieval / pipeline 六大独立模块

## 快速开始

### 前置条件

- Python 3.10+
- SiliconFlow API Key（[前往申请](https://cloud.siliconflow.cn)）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/fanshiqi-source/-rag-rag-.git
cd -rag-rag-

# 2. 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发依赖（测试、代码检查）

# 4. 安装项目包
pip install -e .
```

### 配置 API Key

```bash
# 方式一：环境变量（推荐）
export SILICON_API_KEY=your-api-key-here

# 方式二：.env 文件（从模板创建）
cp .env.example .env
# 编辑 .env 填入真实 Key
```

### 运行

```bash
streamlit run app.py
```

## 项目结构

```
.
├── .github/workflows/ci.yml    # GitHub Actions CI
├── .streamlit/
│   └── secrets.toml             # 密钥文件（已 gitignore）
├── src/nbrag/
│   ├── config.py                # 配置管理
│   ├── llm_client.py            # LLM/Embedding/Rerank API 封装（重试+超时）
│   ├── chunking.py              # 文本切块 + 中文分词
│   ├── indexing.py              # 向量库 + BM25 索引构建
│   ├── retrieval.py             # 基础检索 + RRF 融合
│   └── pipeline.py              # 高级 RAG 流水线
├── tests/                       # 单元测试
├── app.py                       # Streamlit UI
├── requirements.txt             # 生产依赖
├── requirements-dev.txt         # 开发依赖
└── pyproject.toml               # 项目元数据
```

## 运行测试

```bash
pytest -v
```

## CI

项目包含 GitHub Actions CI 配置（`.github/workflows/ci.yml`），每次 push 自动执行：

- Ruff 代码规范检查
- Pytest 单元测试

## 致谢

- [SiliconFlow](https://siliconflow.cn) 提供 LLM / Embedding / Rerank API
- [Milvus Lite](https://milvus.io) 本地向量数据库
- [Streamlit](https://streamlit.io) 交互式 UI 框架
