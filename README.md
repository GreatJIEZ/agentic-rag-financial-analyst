# agentic-rag-financial-analyst
> 一个有限算力下的 Agentic RAG 财报分析助手，基于 Ollama + LangChain 构建，支持在 4GB 显存下高效部署和运行 7B/4B/2B 量化大模型。

## ✨ 项目亮点
一个有限算力情况下的 Agentic RAG 项目，基于 Ollama + LangChain 构建，支持在 4GB 显存下高效部署和运行 Qwen7B/4B/2B 量化模型。通过融合向量检索、BM25 召回、RAG Fusion 等策略，实现对财报 PDF 的精准信息抽取与智能问答，降低幻觉并提升准确率。前后端采用 FastAPI + Gradio 实现，支持内网穿透公网部署，可直接演示与测试。

- **低显存适配**：基于 Ollama 部署 Qwen3.5:4B/2B、Qwen2:7B 等量化模型，完成低显存参数优化与速度/精度对比，实现 4GB 显存下 7B/4B/2B 模型高效推理。
- **混合检索引擎**：构建 Agentic RAG 引擎，实现多类型 PDF 解析、文本清洗、关键词增强与双层精准检索；融合向量检索 + BM25 召回、RAG Fusion、上下文压缩等策略，降低幻觉并提升抽取准确率。
- **完整工程化**：使用 FastAPI + Gradio 搭建前后端服务，通过内网穿透完成公网部署，实现项目可测试与落地。
- **轻松更换不同策略**在api_fastapi.py中将对应的工具进行导入更改即可使用不同的策略来测试效果

## 🛠️ 技术栈
- **大模型部署**：Ollama (Qwen3.5:4B/2B, Qwen2:7B)
- **框架与工具**：LangChain, Chroma, BM25, RAG Fusion
- **服务与部署**：FastAPI, Gradio, cpolar内网穿透
- **开发语言**：Python

## 效果展示
![财报分析助手手机界面](https://raw.githubusercontent.com/GreatJIEZ/agentic-rag-financial-analyst/main/demo1.jpg)
## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/GreatJIEZ/agentic-rag-financial-analyst.git
cd agentic-rag-financial-analyst

# 安装依赖
pip install -r requirements.txt

# 拉取并运行量化模型（可自由选择不同模型）
ollama run qwen2:latest

# 启动 FastAPI 后端
python api_fastapi.py

# 启动 Gradio 前端
python app_gradio.py

