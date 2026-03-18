# 基于 VLLM/Ollama+LangGraph 的多轮记忆财报分析智能问答系统
> 面向有限算力场景的 Agentic RAG 财报分析助手，基于 VLLM/Ollama + LangGraph 构建多智能体协作架构，支持 4GB 显存下高效运行 Qwen7B/4B/2B 量化大模型，新增多轮对话上下文记忆能力，实现财报解析-数据提取-指标计算-专业分析的全流程闭环。

## ✨ 核心亮点
基于 VLLM/Ollama 轻量化部署与 LangGraph 多智能体框架，打造低显存、高鲁棒性的财报智能分析系统，在保留原有精准检索能力的基础上，新增多轮对话记忆与自主决策能力，兼顾效率与实用性：

- **低显存高效推理**：深度优化 Qwen3.5:4B/2B、Qwen2:7B 量化模型部署参数，完成速度/精度对比测试，**实现 4GB 显存下多规格大模型高效推理**，大幅降低硬件部署门槛；
- **多智能体自主决策**：基于 LangGraph 构建数据提取、指标计算、专业分析、反思重试四大核心 Agent，设计重试降级机制，**实现财报分析自主决策闭环**，分析专业度提升 60%+；
- **增强型混合检索 RAG**：升级 Agentic RAG 引擎，融合向量检索 + BM25 召回 + RAG Fusion 策略，优化 PDF 解析、文本清洗与关键词增强逻辑，**显著提升财报信息抽取准确率与检索鲁棒性**，降低模型幻觉；
- **多轮对话上下文记忆**：集成智能截断与对话历史管理策略，支持跨轮指代追问（如“该数值同比增长多少”），**多轮对话响应速度提升 30%+**，1500 字符上下文内精准保留核心财务数据；
- **工程化落地能力**：FastAPI + Gradio 前后端分离架构，支持内网穿透公网部署，可直接上传财报 PDF 完成智能问答/专业分析，**系统有效回答率达 80%+**，支持单/多页财报解析；
- **灵活的策略切换**：在 `api_fastapi.py` 中替换工具导入即可快速切换检索/分析策略，便于对比不同方案效果。

## 🛠️ 技术栈
- **大模型部署**：Ollama (Qwen3.5:4B/2B, Qwen2:7B)/VLLM PyTorch版本: 2.4.1+cu124
- **核心框架**：LangChain, LangGraph, Chroma
- **检索策略**：BM25, 向量检索, RAG Fusion
- **服务与部署**：FastAPI, Gradio, cpolar 内网穿透
- **开发语言**：Python
- **关键能力**：多轮上下文记忆、智能体协作、财务指标自动化计算

## 📸 效果展示
<img src="https://raw.githubusercontent.com/GreatJIEZ/agentic-rag-financial-analyst/main/demo1.jpg" width="500">
> 支持多轮连续追问，自动关联历史对话与财报数据，输出结构化分析结果

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/GreatJIEZ/agentic-rag-financial-analyst.git
cd agentic-rag-financial-analyst

# 安装依赖（含 LangGraph 多智能体与记忆模块）
pip install -r requirements.txt

# 拉取并运行量化模型（按需选择，4GB 显存推荐 qwen2:4b）用ollama可以快速验证效果
ollama run qwen2:4b  # 或 qwen3.5:2b / qwen2:7b

# 启动 FastAPI 后端（含多轮记忆与智能体逻辑）
python api_fastapi.py

# 启动 Gradio 前端（可视化交互界面）
python app_gradio.py
