import os
import re
import shutil
import uuid
from typing import List, Optional
from opencc import OpenCC

import pdfplumber

# LangChain 相关
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ===================== 全局配置（适配多页长财报） =====================
CONFIG = {
    "llm_model": "qwen2",
    "embed_model": "quentinz/bge-small-zh-v1.5:latest",
    "ollama_base_url": "http://localhost:11434",
    "chunk_size": 400,  # 大幅缩减分块（适配长财报）
    "chunk_overlap": 30,  # 最小化重叠（减少冗余）
    "retrieve_k": 2,  # 向量检索只取1条（最相关的）
    "keyword_k": 2,  # 关键词检索只取1条（补充）
    "similarity_threshold": 0.25,  # 提高相似度阈值（只取最相关的）
    "chroma_parent_dir": "./chroma_db_parent",
    "max_single_doc_length": 500,  # 单篇文档最大长度
    "max_total_context_length": 1500,  # 总上下文严格控在1200字符内
}

# 财务关键词扩展（增加更多变体，提升过滤精准度）
FINANCE_KEYS = {
    "净利润": ["归母净利润", "归属于母公司净利润", "净利", "纯利润", "税后利润"],
    "营业收入": ["營收", "营业总收入", "营收", "销售收入", "经营收入"],
    "总资产": ["资产总额", "资产总计", "资产规模"],
    "总负债": ["负债总额", "负债总计", "债务总额"],
    "首席执行官": ["CEO", "行政总裁", "首席执行长"],
    "艺人": ["签约艺人", "旗下艺人", "艺人团队"],
    "高级管理层": ["管理層", "高管团队", "管理层", "高管人员"]
}

cc = OpenCC("t2s")


# ===================== 工具函数（多页财报专属优化） =====================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 清理乱码字符
    garbage_chars = [
        "ྠ", "ᛆ", "ू", "ᛅ", "ᑛ", "ϗ", "ʮ", "ሜ", "ື", "΢",
        "ॹ", "௅", "ʱ", "᙮", "Ꮄ", "΋", "ୃ", "ኽ", "ྼ", "᜗",
        "І", "Ԓ", "ʈ", "Ո", "༈", "ഃ", "̙", "ᔷ", "आ", "૿",
        "Υ", "͊", "຾", "ᄲ", "ᜊ", "ਗ", "Ԩ", "ʫ", "Ϟ", "Դ", "౬", "ව",
        "ϋ", "జ", "ੂ", "Б", "໨", "ԫ", "Ӂ", "ശ", "ɾ", "ɻ", "Τ", "ᙧ", "ᔖ", "З", "މ", "ග"
    ]
    for c in garbage_chars:
        text = text.replace(c, "")

    # 清理控制字符+保留核心字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,%\(\)\+\-\*\/\:\;\"\'\¥\$]', '', text)

    # 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    # 繁转简
    try:
        text = cc.convert(text)
    except:
        pass

    return text


def smart_truncate(text: str, max_len: int, keyword: str = "") -> str:
    """
    针对财务问题的智能截断：
    1. 优先保留含关键词的段落
    2. 优先保留数字和财务术语
    3. 截断后保证语义完整
    """
    if len(text) <= max_len:
        return text

    # 步骤1：如果有关键词，先提取含关键词的句子
    if keyword:
        # 按句子分割
        sentences = re.split(r'[。！？；]', text)
        key_sentences = [s for s in sentences if keyword in s]
        if key_sentences:
            text = "。".join(key_sentences)[:max_len - 50]  # 留50字符给数字

    # 步骤2：提取数字和核心财务术语（保证关键数据不丢）
    finance_terms = "|".join([k for v in FINANCE_KEYS.values() for k in v] + list(FINANCE_KEYS.keys()))
    key_info = re.findall(rf'({finance_terms}|\d+[\.,]?\d*%?|\d+万|\d+亿)', text)
    key_str = " 【关键数据】：" + " ".join(list(set(key_info)))[:100]  # 去重+控长

    # 步骤3：截断主体+补充关键信息
    main_text = text[:max_len - len(key_str) - 10].strip()
    final_text = main_text + key_str

    return final_text[:max_len]


def filter_by_keyword(text: str, keywords: List[str]) -> str:
    """
    只保留含指定关键词的内容（过滤无关文本，大幅减长度）
    """
    if not keywords or not text:
        return text

    # 按行分割，只保留含关键词的行
    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        if any(kw in line for kw in keywords):
            filtered_lines.append(line.strip())

    # 如果过滤后无内容，返回前200字符（兜底）
    if not filtered_lines:
        return text[:200]

    return " ".join(filtered_lines)


def load_pdf_with_pdfplumber(pdf_path: str) -> List[Document]:
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    # 加载时就初步过滤（只保留有财务术语的页）
                    finance_terms = "|".join([k for v in FINANCE_KEYS.values() for k in v] + list(FINANCE_KEYS.keys()))
                    if re.search(finance_terms, page_text):
                        documents.append(Document(
                            page_content=page_text,
                            metadata={"page": page_num}
                        ))
        print(f"📌 加载PDF完成，过滤后有效页数：{len(documents)}（原{len(pdf.pages)}页）")
        return documents
    except Exception as e:
        print(f"⚠️ PDF加载失败：{e}")
        return []


def clean_old_chroma_dirs(parent_dir: str, max_keep: int = 5):
    if not os.path.exists(parent_dir):
        return
    dirs = []
    for item in os.listdir(parent_dir):
        p = os.path.join(parent_dir, item)
        if os.path.isdir(p):
            dirs.append((p, os.path.getctime(p)))
    dirs.sort(key=lambda x: x[1], reverse=True)
    for d, _ in dirs[max_keep:]:
        try:
            shutil.rmtree(d)
        except:
            pass


# ===================== FinanceRAG 主体（多页财报专属） =====================
class FinanceRAG:
    def __init__(self):
        # 强制限制LLM上下文窗口（适配Qwen2）
        self.llm = OllamaLLM(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.1,
            num_ctx=2048,  # Qwen2-7B的上下文上限
            max_tokens=512  # 限制生成回答长度，留更多空间给输入
        )
        self.embeddings = OllamaEmbeddings(
            model=CONFIG["embed_model"],
            base_url=CONFIG["ollama_base_url"]
        )
        self.db = None
        self.current_db_path = None
        self.bm25_retriever = None
        self.split_docs = []

    def build_db(self, pdf_path: str):
        if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
            raise Exception("文件无效或不是PDF")

        os.makedirs(CONFIG["chroma_parent_dir"], exist_ok=True)
        self.current_db_path = os.path.join(CONFIG["chroma_parent_dir"], f"chroma_db_{uuid.uuid4().hex[:8]}")
        clean_old_chroma_dirs(CONFIG["chroma_parent_dir"])

        # 加载PDF时已过滤无关页（大幅减少数据量）
        raw_docs = load_pdf_with_pdfplumber(pdf_path)
        if not raw_docs:
            raise Exception("PDF无有效财务文本内容")

        # 清洗文本
        cleaned_docs = []
        for doc in raw_docs:
            t = clean_text(doc.page_content)
            if len(t) > 10:
                cleaned_docs.append(Document(page_content=t, metadata=doc.metadata))

        # 小尺寸分块（适配长财报）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]
        )
        self.split_docs = splitter.split_documents(cleaned_docs)
        print(f"📌 文本分块完成，有效分块数：{len(self.split_docs)}（小尺寸分块）")

        # 关键词检索器只取1条
        self.bm25_retriever = BM25Retriever.from_documents(self.split_docs)
        self.bm25_retriever.k = CONFIG["keyword_k"]

        # 构建向量库
        self.db = Chroma.from_documents(
            documents=self.split_docs,
            embedding=self.embeddings,
            persist_directory=self.current_db_path
        )
        print("✅ 向量库构建完成（适配多页长财报）")

    def hybrid_retrieval(self, query: str, filter_keywords: List[str]) -> List[Document]:
        """
        混合检索+关键词过滤（核心：只取最相关的）
        """
        query = cc.convert(query)

        # 1. 向量检索只取1条最相关的
        vec_docs_w_score = self.db.similarity_search_with_score(query, k=CONFIG["retrieve_k"])
        vec_docs = [d for d, s in vec_docs_w_score if s < CONFIG["similarity_threshold"]]
        if not vec_docs:
            vec_docs = [d for d, _ in vec_docs_w_score[:1]]  # 兜底只取1条

        # 2. 关键词检索只取1条
        kw_docs = self.bm25_retriever.get_relevant_documents(query)[:CONFIG["keyword_k"]]

        # 3. 合并+去重+关键词过滤（核心减长度）
        combined_docs = []
        seen = set()
        for doc in vec_docs + kw_docs:
            # 先过滤：只保留含关键词的文档
            filtered_content = filter_by_keyword(doc.page_content, filter_keywords)
            if not filtered_content:
                continue

            # 去重
            key = (filtered_content[:50], doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                # 替换为过滤后的内容
                filtered_doc = Document(
                    page_content=filtered_content,
                    metadata=doc.metadata
                )
                combined_docs.append(filtered_doc)

        # 最终只保留最多2条（1向量+1关键词）
        return combined_docs[:CONFIG["retrieve_k"] + CONFIG["keyword_k"]]

    def query(self, question: str) -> str:
        if not self.db:
            raise Exception("请先执行build_db加载PDF")

        # 1. 提取问题关键词（核心：只聚焦当前问题）
        target_key = None
        filter_keywords = []
        for k, aliases in FINANCE_KEYS.items():
            if k in question:
                target_key = k
                filter_keywords = [k] + aliases
                break
        # 兜底：如果无匹配关键词，用问题本身作为关键词
        if not filter_keywords:
            filter_keywords = [question[:10]]

        # 2. 混合检索（只取最相关的，且过滤无关内容）
        docs = self.hybrid_retrieval(question, filter_keywords)
        if not docs:
            return "未找到相关数据\n数据来源：无"

        # 3. 智能截断（核心控长，保关键信息）
        processed_docs = []
        total_length = 0
        pages = []
        for doc in docs:
            # 单文档截断到500字符内
            truncated_content = smart_truncate(
                doc.page_content,
                CONFIG["max_single_doc_length"],
                target_key or ""
            )
            # 控制总长度
            if total_length + len(truncated_content) > CONFIG["max_total_context_length"]:
                remaining = CONFIG["max_total_context_length"] - total_length
                if remaining > 100:  # 至少保留100字符
                    truncated_content = truncated_content[:remaining]
                    processed_docs.append(truncated_content)
                    total_length += len(truncated_content)
                break
            processed_docs.append(truncated_content)
            total_length += len(truncated_content)
            pages.append(str(doc.metadata.get("page", "未知")))

        # 4. 构建上下文（严格控长）
        context = "\n---\n".join(processed_docs)
        print(f"\n🔍 最终上下文（长度：{len(context)}字符）：")
        print(f"   过滤后仅保留和「{target_key or question[:10]}」相关的内容")
        for i, p in enumerate(pages):
            preview = processed_docs[i][:80] + "..." if len(processed_docs[i]) > 80 else processed_docs[i]
            print(f"   第{p}页：{preview}")

        # 5. 构建极简提示词（进一步减长度）
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
规则：
1. 仅回答问题：{question}
2. 仅使用上下文数据，不编造
3. 无数据输出：未找到相关数据

上下文：
{context}

答案：
"""
        )

        # 6. LLM推理（长度已严格控制）
        try:
            chain = prompt_template | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question
            }).strip()
        except Exception as e:
            raise Exception(f"❌ 回答生成失败：{str(e)}")

        # 7. 格式化回答
        return f"{answer}\n上下文数据来源：第{','.join(pages)}页"


# ===================== 测试（多页财报专用） =====================
if __name__ == "__main__":
    rag = FinanceRAG()
    # 替换为你的多页财报PDF路径
    pdf_file_path = r"E:\conda\envs\llm_1_9\乐华年报.pdf"

    try:
        rag.build_db(pdf_file_path)
    except Exception as e:
        print("❌ 向量库构建失败：", e)
        exit(1)

    # 测试多页财报查询
    test_questions = [
        "首席执行官是？",
        "高级管理层有？",
        "营业收入是多少？",
        "净利润是多少？",
        "总资产是多少？",
        "总负债是多少？",
        "杜华女士在哪个学校毕业的？"
    ]

    print("\n" + "=" * 80)
    print("📊 多页财报问答测试（严格控长+精准聚焦）")
    print("=" * 80)
    for q in test_questions:
        print(f"\n📝 问题：{q}")
        try:
            ans = rag.query(q)
            print(f"✅ 回答：{ans}")
        except Exception as e:
            print(f"❌ 错误：{e}")