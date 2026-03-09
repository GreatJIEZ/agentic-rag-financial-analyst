import os
import re
import shutil
import uuid
import json
from typing import List, Optional, Dict, TypedDict
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

# LangGraph 相关导入
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# ===================== 全局配置 =====================
CONFIG = {
    "llm_model": "qwen2",
    "embed_model": "quentinz/bge-small-zh-v1.5:latest",
    "ollama_base_url": "http://localhost:11434",
    "chunk_size": 400,
    "chunk_overlap": 30,
    "retrieve_k": 2,
    "keyword_k": 2,
    "similarity_threshold": 0.25,
    "chroma_parent_dir": "./chroma_db_parent",
    "max_single_doc_length": 500,
    "max_total_context_length": 1500,
    "max_retry_count": 3,
}

# 财务关键词扩展
FINANCE_KEYS = {
    "净利润": ["归母净利润", "归属于母公司净利润", "净利", "纯利润", "税后利润"],
    "营业收入": ["營收", "营业总收入", "营收", "销售收入", "经营收入"],
    "总资产": ["资产总额", "资产总计", "资产规模"],
    "总负债": ["负债总额", "负债总计", "债务总额"],
    "首席执行官": ["CEO", "行政总裁", "首席执行长"],
    "艺人": ["签约艺人", "旗下艺人", "艺人团队"],
    "高级管理层": ["管理層", "高管团队", "管理层", "高管人员"]
}

# 财务指标计算映射
FINANCE_CALC_MAP = {
    "毛利率": {"分子": "净利润", "分母": "营业收入", "公式": "净利润/营业收入*100%"},
    "资产负债率": {"分子": "总负债", "分母": "总资产", "公式": "总负债/总资产*100%"},
    "营收增长率": {"分子": "当期营收", "分母": "上期营收", "公式": "(当期营收-上期营收)/上期营收*100%"},
}

cc = OpenCC("t2s")


# ===================== 工具函数=====================
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
    if len(text) <= max_len:
        return text

    # 步骤1：提取含关键词的句子
    if keyword:
        sentences = re.split(r'[。！？；]', text)
        key_sentences = [s for s in sentences if keyword in s]
        if key_sentences:
            text = "。".join(key_sentences)[:max_len - 50]

    # 步骤2：提取数字和财务术语
    finance_terms = "|".join([k for v in FINANCE_KEYS.values() for k in v] + list(FINANCE_KEYS.keys()))
    key_info = re.findall(rf'({finance_terms}|\d+[\.,]?\d*%?|\d+万|\d+亿)', text)
    key_str = " 【关键数据】：" + " ".join(list(set(key_info)))[:100]

    # 步骤3：截断主体+补充关键信息
    main_text = text[:max_len - len(key_str) - 10].strip()
    final_text = main_text + key_str

    return final_text[:max_len]


def filter_by_keyword(text: str, keywords: List[str]) -> str:
    if not keywords or not text:
        return text

    lines = text.split("\n")
    filtered_lines = []
    for line in lines:
        if any(kw in line for kw in keywords):
            filtered_lines.append(line.strip())

    if not filtered_lines:
        return text[:200]

    return " ".join(filtered_lines)


# 提取财务数值
def extract_financial_number(text: str) -> float:
    num_match = re.search(r'(\d+[\.,]?\d*)', text)
    if not num_match:
        return 0.0

    num = float(num_match.group(1).replace(",", ""))
    if "亿" in text:
        num *= 100000000
    elif "万" in text:
        num *= 10000
    return num


def load_pdf_with_pdfplumber(pdf_path: str) -> List[Document]:
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
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


# ===================== LangGraph 状态定义 =====================
class AnalysisState(TypedDict):
    """智能体全局状态（所有字段均为必填）"""
    question: str
    rag_instance: object
    raw_data: Dict[str, str]
    calculated_indicators: Dict[str, str]
    analysis_result: str
    error: str
    retry_count: int
    pages: List[str]


# ===================== FinanceRAG 核心类 =====================
class FinanceRAG:
    def __init__(self):
        self.llm = OllamaLLM(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.1,
            num_ctx=2048,
            max_tokens=512
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

        raw_docs = load_pdf_with_pdfplumber(pdf_path)
        if not raw_docs:
            raise Exception("PDF无有效财务文本内容")

        cleaned_docs = []
        for doc in raw_docs:
            t = clean_text(doc.page_content)
            if len(t) > 10:
                cleaned_docs.append(Document(page_content=t, metadata=doc.metadata))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]
        )
        self.split_docs = splitter.split_documents(cleaned_docs)
        print(f"📌 文本分块完成，有效分块数：{len(self.split_docs)}（小尺寸分块）")

        self.bm25_retriever = BM25Retriever.from_documents(self.split_docs)
        self.bm25_retriever.k = CONFIG["keyword_k"]

        self.db = Chroma.from_documents(
            documents=self.split_docs,
            embedding=self.embeddings,
            persist_directory=self.current_db_path
        )
        print("✅ 向量库构建完成")

    def hybrid_retrieval(self, query: str, filter_keywords: List[str]) -> List[Document]:
        query = cc.convert(query)

        # 向量检索
        vec_docs_w_score = self.db.similarity_search_with_score(query, k=CONFIG["retrieve_k"])
        vec_docs = [d for d, s in vec_docs_w_score if s < CONFIG["similarity_threshold"]]
        if not vec_docs:
            vec_docs = [d for d, _ in vec_docs_w_score[:1]]

        # 关键词检索
        kw_docs = self.bm25_retriever.get_relevant_documents(query)[:CONFIG["keyword_k"]]

        # 合并+去重+过滤
        combined_docs = []
        seen = set()
        for doc in vec_docs + kw_docs:
            filtered_content = filter_by_keyword(doc.page_content, filter_keywords)
            if not filtered_content:
                continue

            key = (filtered_content[:50], doc.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                filtered_doc = Document(
                    page_content=filtered_content,
                    metadata=doc.metadata
                )
                combined_docs.append(filtered_doc)

        return combined_docs[:CONFIG["retrieve_k"] + CONFIG["keyword_k"]]

    # 提取上下文（供智能体调用）
    def get_context(self, question: str) -> tuple[str, List[str]]:
        # 提取关键词
        target_key = None
        filter_keywords = []
        for k, aliases in FINANCE_KEYS.items():
            if k in question:
                target_key = k
                filter_keywords = [k] + aliases
                break
        if not filter_keywords:
            filter_keywords = [question[:10]]

        # 混合检索
        docs = self.hybrid_retrieval(question, filter_keywords)
        if not docs:
            return "", []

        # 智能截断
        processed_docs = []
        total_length = 0
        pages = []
        for doc in docs:
            truncated_content = smart_truncate(
                doc.page_content,
                CONFIG["max_single_doc_length"],
                target_key or ""
            )
            if total_length + len(truncated_content) > CONFIG["max_total_context_length"]:
                remaining = CONFIG["max_total_context_length"] - total_length
                if remaining > 100:
                    truncated_content = truncated_content[:remaining]
                    processed_docs.append(truncated_content)
                    total_length += len(truncated_content)
                break
            processed_docs.append(truncated_content)
            total_length += len(truncated_content)
            pages.append(str(doc.metadata.get("page", "未知")))

        context = "\n---\n".join(processed_docs)
        return context, pages

    def query(self, question: str) -> str:
        if not self.db:
            raise Exception("请先执行build_db加载PDF")

        context, pages = self.get_context(question)
        if not context:
            return "未找到相关数据\n数据来源：无"

        # 构建提示词
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

        # LLM推理
        try:
            chain = prompt_template | self.llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "question": question
            }).strip()
        except Exception as e:
            raise Exception(f"❌ 回答生成失败：{str(e)}")

        return f"{answer}\n上下文数据来源：第{','.join(pages)}页"

    # ===================== 智能体方法（核心修复） =====================
    def extract_agent(self, state: AnalysisState) -> AnalysisState:
        """数据提取智能体"""
        try:
            question = state["question"]
            rag = state["rag_instance"]

            raw_data = {}
            pages = []
            for key in FINANCE_KEYS.keys():
                context, key_pages = rag.get_context(key)
                if context:
                    prompt = PromptTemplate(
                        input_variables=["context", "key"],
                        template="""
从以下上下文中提取{key}的具体数值（仅返回数值+单位，如"100亿"或"5000万元"）：
{context}
"""
                    )
                    chain = prompt | self.llm | StrOutputParser()
                    value = chain.invoke({"context": context, "key": key}).strip()
                    if value and "未找到" not in value:
                        raw_data[key] = value
                        pages.extend(key_pages)

            pages = list(set(pages))
            # 核心修复：继承全量状态字段
            return {
                **state,
                "raw_data": raw_data,
                "pages": pages,
                "error": "",
                "retry_count": 0
            }
        except Exception as e:
            # 核心修复：继承全量状态字段
            return {**state, "error": f"提取失败：{str(e)}", "retry_count": state["retry_count"] + 1}

    def calculate_agent(self, state: AnalysisState) -> AnalysisState:
        """指标计算智能体"""
        if state.get("error") or not state["raw_data"]:
            return state  # 直接返回原状态，避免字段丢失

        try:
            raw_data = state["raw_data"]
            calculated = {}

            # 计算毛利率
            if "净利润" in raw_data and "营业收入" in raw_data:
                profit_num = extract_financial_number(raw_data["净利润"])
                revenue_num = extract_financial_number(raw_data["营业收入"])
                if revenue_num > 0:
                    gross_margin = (profit_num / revenue_num) * 100
                    calculated["毛利率"] = f"{gross_margin:.2f}%"

            # 计算资产负债率
            if "总负债" in raw_data and "总资产" in raw_data:
                debt_num = extract_financial_number(raw_data["总负债"])
                asset_num = extract_financial_number(raw_data["总资产"])
                if asset_num > 0:
                    debt_ratio = (debt_num / asset_num) * 100
                    calculated["资产负债率"] = f"{debt_ratio:.2f}%"

            # 核心修复：继承全量状态字段
            return {**state, "calculated_indicators": calculated, "error": ""}
        except Exception as e:
            # 核心修复：继承全量状态字段
            return {**state, "error": f"计算失败：{str(e)}", "retry_count": state["retry_count"] + 1}

    def analyze_agent(self, state: AnalysisState) -> AnalysisState:
        """分析总结智能体"""
        if state.get("error") or not state["raw_data"]:
            return state

        try:
            prompt_template = PromptTemplate(
                input_variables=["raw_data", "calculated", "question"],
                template="""
基于以下财务数据，专业分析回答用户问题：{question}

### 原始财务数据
{raw_data}

### 计算指标
{calculated}

### 分析要求
1. 数据准确，基于提供的财务数据
2. 结论专业，符合财务分析规范
3. 指出核心亮点/风险（如有）
4. 语言简洁，逻辑清晰
"""
            )

            chain = prompt_template | self.llm | StrOutputParser()
            analysis = chain.invoke({
                "raw_data": json.dumps(state["raw_data"], ensure_ascii=False, indent=2),
                "calculated": json.dumps(state["calculated_indicators"], ensure_ascii=False, indent=2),
                "question": state["question"]
            }).strip()

            # 核心修复：继承全量状态字段
            return {**state, "analysis_result": analysis, "error": ""}
        except Exception as e:
            # 核心修复：继承全量状态字段
            return {**state, "error": f"分析失败：{str(e)}", "retry_count": state["retry_count"] + 1}

    def reflect_agent(self, state: AnalysisState) -> AnalysisState:
        """反思智能体"""
        # 核心修复：继承全量状态字段
        if state["retry_count"] >= CONFIG["max_retry_count"]:
            return {**state, "error": "重试次数超限，返回现有结果"}

        if not state["raw_data"]:
            return {**state, "error": "未提取到有效财务数据，需要重试"}
        elif any(indicator in state["question"] for indicator in ["毛利率", "资产负债率"]) and not state[
            "calculated_indicators"]:
            return {**state, "error": "关键指标计算失败，需要重试"}
        else:
            return {**state, "error": ""}

    def build_analysis_agent(self) -> CompiledStateGraph:
        """构建LangGraph工作流"""
        graph = StateGraph(AnalysisState)

        # 添加节点
        graph.add_node("extract", self.extract_agent)
        graph.add_node("calculate", self.calculate_agent)
        graph.add_node("analyze", self.analyze_agent)
        graph.add_node("reflect", self.reflect_agent)

        # 条件边函数
        def should_retry(state: AnalysisState) -> str:
            if state.get("error") and state["retry_count"] < CONFIG["max_retry_count"]:
                return "extract"
            else:
                return END

        # 定义工作流
        graph.set_entry_point("extract")
        graph.add_edge("extract", "calculate")
        graph.add_edge("calculate", "analyze")
        graph.add_edge("analyze", "reflect")
        graph.add_conditional_edges("reflect", should_retry, {"extract": "extract", END: END})

        return graph.compile()

    def analyze(self, question: str) -> str:
        """智能体分析入口"""
        if not self.db:
            raise Exception("请先执行build_db加载PDF")

        analysis_graph = self.build_analysis_agent()

        # 初始状态（包含所有必填字段）
        initial_state = {
            "question": question,
            "rag_instance": self,
            "raw_data": {},
            "calculated_indicators": {},
            "analysis_result": "",
            "error": "",
            "retry_count": 0,
            "pages": []
        }

        result = analysis_graph.invoke(initial_state)

        # 格式化结果
        if result.get("error"):
            base_answer = self.query(question)
            return f"⚠️ 智能分析失败：{result['error']}\n\n基础回答：{base_answer}"

        if not result["analysis_result"]:
            return self.query(question)

        final_answer = f"""
📊 财报专业分析结果
====================
用户问题：{question}

### 核心财务数据
{json.dumps(result['raw_data'], ensure_ascii=False, indent=2)}

### 计算指标
{json.dumps(result['calculated_indicators'], ensure_ascii=False, indent=2)}

### 专业分析
{result['analysis_result']}

📋 数据来源：第{','.join(result['pages'])}页
"""
        return final_answer


# ===================== 测试 =====================
if __name__ == "__main__":
    # 初始化并加载PDF
    rag = FinanceRAG()
    pdf_file_path = r"E:\conda\envs\llm_1_9\乐华年报.pdf"

    try:
        rag.build_db(pdf_file_path)
    except Exception as e:
        print("❌ 向量库构建失败：", e)
        exit(1)

    # 测试1：基础问答
    print("\n" + "=" * 80)
    print("📊 基础问答测试")
    print("=" * 80)
    basic_questions = [
        "首席执行官是？",
        "营业收入是多少？",
        "净利润是多少？"
    ]
    for q in basic_questions:
        print(f"\n📝 问题：{q}")
        try:
            ans = rag.query(q)
            print(f"✅ 回答：{ans}")
        except Exception as e:
            print(f"❌ 错误：{e}")

    # 测试2：智能体分析
    print("\n" + "=" * 80)
    print("📊 财报智能体专业分析")
    print("=" * 80)
    analysis_questions = [
        "分析这家公司的财务健康状况",
        "计算这家公司的毛利率和资产负债率",
        "这家公司的偿债能力如何？"
    ]
    for q in analysis_questions:
        print(f"\n📝 分析问题：{q}")
        try:
            ans = rag.analyze(q)
            print(f"✅ 分析结果：{ans}")
        except Exception as e:
            print(f"❌ 错误：{e}")
