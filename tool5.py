# 相比tool3增加了RAG Fusion + 父块检索 + 上下文压缩
import os
import re
import shutil
import time
import uuid
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict

# LangChain 核心导入
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

# ===================== 全局配置=====================
CONFIG = {
    "llm_model": "qwen2",
    "embed_model": "quentinz/bge-small-zh-v1.5:latest",
    "ollama_base_url": "http://localhost:11434",
    "child_chunk_size": 256,
    "child_chunk_overlap": 30,
    "parent_chunk_size": 800,
    "parent_chunk_overlap": 50,
    "retrieve_k": 2,
    "fusion_query_num": 3,
    "similarity_threshold": 0.35,
    "chroma_parent_dir": "./chroma_db_parent",
    "max_context_length": 1024,
}

# 财务关键词映射
FINANCE_KEYS = {
    "净利润": ["淨利潤", "归母净利润", "归属于母公司净利润"],
    "营业收入": ["營業收入", "營收", "营业总收入"],
    "总资产": ["總資產", "资产总额", "资产总计"],
    "总负债": ["總負債", "负债总额", "负债总计"],
    "首席执行官": ["首席執行官", "CEO", "行政总裁"],
    "艺人": ["藝人", "签约艺人", "旗下艺人"],
    "高级管理层": ["高級管理層", "管理層", "高管团队"]
}

# RAG Fusion 提示词
FUSION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
为问题【{question}】生成{num}个同义改写问题，只改写措辞，不改变核心含义，每个问题不超过20个字，只输出问题，每行一个。
"""
)


# ===================== 工具函数 =====================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 移除乱码
    garbage_chars = ["ϋϋజ", "ੂБ໨ԫ", "Ӂശɾɻ", "Τ ϋᙧ", "ᔖЗމග", "\u000b", "\u000c", "\u000e", "\u000f"]
    for char in garbage_chars:
        text = text.replace(char, "")

    # 清理空格+控制长度
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:CONFIG["max_context_length"]]


def clean_old_chroma_dirs(parent_dir: str, max_keep: int = 3) -> None:
    """清理旧向量库目录"""
    if not os.path.exists(parent_dir):
        return
    dirs = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            dirs.append((item_path, os.path.getctime(item_path)))

    dirs.sort(key=lambda x: x[1], reverse=True)
    for dir_path, _ in dirs[max_keep:]:
        try:
            shutil.rmtree(dir_path)
        except:
            pass


def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List:
    """RAG Fusion核心算法"""
    fused_scores = defaultdict(float)
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = f"{doc.metadata.get('page', 'unknown')}_{doc.page_content[:50]}"
            fused_scores[doc_id] += 1 / (rank + k)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    seen = set()
    final_docs = []
    for doc_id, _ in sorted_docs:
        for docs in results:
            for doc in docs:
                current_id = f"{doc.metadata.get('page', 'unknown')}_{doc.page_content[:50]}"
                if current_id == doc_id and current_id not in seen:
                    seen.add(current_id)
                    final_docs.append(doc)
                    break
        if len(final_docs) >= CONFIG["retrieve_k"]:
            break
    return final_docs


def lightweight_compression(docs, question):
    """轻量压缩逻辑"""
    keywords = [word for word in question.split() if len(word) > 1]
    if not keywords:
        keywords = list(question)

    compressed_docs = []
    for doc in docs:
        if any(key in doc.page_content for key in keywords):
            doc.page_content = doc.page_content[:300]  # 控制长度
            compressed_docs.append(doc)
    return compressed_docs[:CONFIG["retrieve_k"]]


# ===================== 核心RAG类=====================
class FinanceRAG:
    def __init__(self):
        # 初始化LLM
        self.llm = OllamaLLM(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.0,
            num_ctx=CONFIG["max_context_length"]
        )

        # 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(
            model=CONFIG["embed_model"],
            base_url=CONFIG["ollama_base_url"]
        )

        # 检索器初始化
        self.parent_retriever = None
        self.bm25_retriever = None
        self.db = None
        self.store = InMemoryStore()
        self.current_db_path = None

    def build_db(self, pdf_path: str) -> None:
        """构建向量库和检索器"""
        # 基础校验
        if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
            raise Exception("❌ PDF文件不存在或格式错误")

        # 目录准备
        os.makedirs(CONFIG["chroma_parent_dir"], exist_ok=True)
        self.current_db_path = os.path.join(CONFIG["chroma_parent_dir"], f"chroma_db_{uuid.uuid4().hex[:8]}")
        clean_old_chroma_dirs(CONFIG["chroma_parent_dir"])

        # 加载并清洗PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        cleaned_docs = []
        for idx, doc in enumerate(docs):
            cleaned_text = clean_text(doc.page_content)
            if cleaned_text and len(cleaned_text) > 10:
                cleaned_docs.append(Document(page_content=cleaned_text, metadata={"page": idx + 1}))

        if not cleaned_docs:
            raise Exception("❌ PDF清洗后无有效内容")

        # 初始化分块器
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["child_chunk_size"],
            chunk_overlap=CONFIG["child_chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["parent_chunk_size"],
            chunk_overlap=CONFIG["parent_chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]
        )

        # 构建父块检索器
        self.db = Chroma(embedding_function=self.embeddings, persist_directory=self.current_db_path)
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.db,
            docstore=self.store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": CONFIG["retrieve_k"] * 2}
        )
        self.parent_retriever.add_documents([doc for doc in loader.load() if len(clean_text(doc.page_content)) > 10])

        # 初始化BM25检索器
        all_docs = parent_splitter.split_documents(cleaned_docs)
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = CONFIG["retrieve_k"]

        print(f"✅ 向量库构建完成，有效文档数：{len(cleaned_docs)}")

    def generate_fusion_queries(self, question: str) -> List[str]:
        """生成RAG Fusion同义问题"""
        fusion_chain = FUSION_PROMPT.partial(num=CONFIG["fusion_query_num"]) | self.llm | StrOutputParser()
        try:
            fusion_text = fusion_chain.invoke({"question": question})
            fusion_queries = [q.strip() for q in fusion_text.split("\n") if q.strip()]
            fusion_queries = list(dict.fromkeys(fusion_queries))[:CONFIG["fusion_query_num"]]
            return fusion_queries if fusion_queries else [question]
        except:
            return [question]

    def hybrid_fusion_retrieval(self, question: str) -> List[Document]:
        """核心检索逻辑：RAG Fusion + 父块 + BM25 + 轻量压缩"""
        # 生成Fusion查询
        fusion_queries = self.generate_fusion_queries(question)

        # 执行检索
        fusion_results = []
        for query in fusion_queries:
            try:
                vector_docs = self.parent_retriever.get_relevant_documents(query)
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                fusion_results.append(vector_docs + bm25_docs)
            except:
                continue

        # RRF融合
        if fusion_results:
            retrieved_docs = reciprocal_rank_fusion(fusion_results)
        else:
            retrieved_docs = self.parent_retriever.get_relevant_documents(question)

        # 轻量压缩
        final_docs = lightweight_compression(retrieved_docs, question)
        return final_docs

    def query(self, question: str) -> str:
        """问答主函数"""
        if not self.parent_retriever or not question.strip():
            raise Exception("❌ 检索器未初始化或问题为空")

        # 关键词扩展
        target_key = None
        expand_keys = []
        for key, aliases in FINANCE_KEYS.items():
            if key in question:
                target_key = key
                expand_keys = aliases
                break
        expanded_query = question + " " + " ".join(expand_keys)

        # 检索
        retrieved_docs = self.hybrid_fusion_retrieval(expanded_query)
        if not retrieved_docs:
            return "未找到相关数据\n数据来源：无"

        # 过滤关键词文档
        if target_key:
            retrieved_docs = [doc for doc in retrieved_docs if target_key in doc.page_content] or retrieved_docs

        # 构建上下文
        page_numbers = [str(doc.metadata.get("page", "未知")) for doc in retrieved_docs]
        context_text = "\n---\n".join([clean_text(doc.page_content) for doc in retrieved_docs])
        context_text = context_text[:CONFIG["max_context_length"]]

        # 生成回答,可以在这里对模型进行提示词约束
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
严格遵守规则回答问题【{question}】：
1. 仅使用上下文信息，不编造、不扩展
2. 只输出最终答案，无多余文字
3. 找不到内容时输出：未找到相关数据

上下文：{context}
答案：
"""
        )

        chain = prompt_template | self.llm | StrOutputParser()
        raw_answer = chain.invoke({"context": context_text, "question": question}).strip()
        return f"{raw_answer}\n数据来源：第{','.join(page_numbers)}页"


# ===================== 测试入口 =====================
if __name__ == "__main__":
    rag_pro = FinanceRAG()

    # 替换为你的PDF路径
    pdf_path = r"E:\conda\envs\llm_1_9\乐华年报.pdf"
    try:
        rag_pro.build_db(pdf_path)
    except Exception as e:
        print(f"❌ 构建失败：{e}")
        exit(1)

    # 测试问题
    test_questions = [
        "首席执行官是？",
        "营业收入是多少？",
        "净利润是多少？",
        "杜华的职位是什么？"
    ]

    print("=" * 80)
    for q in test_questions:
        print(f"\n🔍 问题：{q}")
        try:
            print(f"✅ 回答：{rag_pro.query(q)}")
        except Exception as e:
            print(f"❌ 回答失败：{e}")
    print("=" * 80)