# 相比tool3增添了混合检索模块，即采用向量为主，BM25关键词为辅的检索
import os
import re
import shutil
import uuid
from typing import List,  Optional


# LangChain 相关导入
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever  # 关键词检索器

# ===================== 全局配置 =====================
CONFIG = {
    "llm_model": "qwen2",
    "embed_model": "quentinz/bge-small-zh-v1.5:latest",  # 专门选的轻量化嵌入模型
    "ollama_base_url": "http://localhost:11434",  # Ollama 服务地址
    "chunk_size": 450,  # 文本分块大小
    "chunk_overlap": 40,  # 分块重叠长度
    "retrieve_k": 3,  # 向量检索相似文本数量
    "keyword_k": 2,  # 关键词检索文本数量
    "similarity_threshold": 0.3,  # 向量相似度阈值（得分越低越相似）
    "chroma_parent_dir": "./chroma_db_parent",  # 统一的向量库父目录
}

# 财务关键词映射（兼容繁体/别名）
FINANCE_KEYS = {
    "净利润": ["淨利潤", "归母净利润", "归属于母公司净利润"],
    "营业收入": ["營業收入", "營收", "营业总收入"],
    "总资产": ["總資產", "资产总额", "资产总计"],
    "总负债": ["總負債", "负债总额", "负债总计"],
    "首席执行官": ["首席執行官", "CEO", "行政总裁"],
    "艺人": ["藝人", "签约艺人", "旗下艺人"],
    "高级管理层": ["高級管理層", "管理層", "高管团队"]
}


# ===================== 工具函数 =====================
def clean_text(text: str) -> str:
    """
    移除乱码、统一繁简体、清理多余空格
    """
    if not isinstance(text, str):
        return ""

    # 移除常见乱码字符
    garbage_chars = [
        "ϋϋజ", "ੂБ໨ԫ", "Ӂശɾɻ", "Τ ϋᙧ", "ᔖЗމග",
        "\u000b", "\u000c", "\u000e", "\u000f",  # 特殊控制字符
        "ੂ", "Б", "໨", "ԫ", "Ӂ", "ശ", "ɾ", "ɻ", "Τ", "ᙧ", "ᔖ", "З", "މ", "ග"  # 单个乱码
    ]
    for char in garbage_chars:
        text = text.replace(char, "")

    # 清理多余空格/换行
    text = re.sub(r'\s+', ' ', text).strip()

    # 繁简转换（对部分财务术语）
    trad2sim_map = {
        '資': '资', '產': '产', '營': '营', '淨': '净', '潤': '润',
        '總': '总', '額': '额', '債': '债', '華': '华', '務': '务',
        '執': '执', '長': '长', '級': '级', '團': '团', '藝': '艺', '員': '员'
    }
    for trad_char, sim_char in trad2sim_map.items():
        text = text.replace(trad_char, sim_char)

    return text


def clean_old_chroma_dirs(parent_dir: str, max_keep: int = 5) -> None:
    """
    :param parent_dir: 向量库父目录
    :param max_keep: 保留最新的N个目录，其余删除
    """
    if not os.path.exists(parent_dir):
        return

    # 获取所有子目录，按创建时间排序（最新在前）
    dirs = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # 获取目录创建时间
            create_time = os.path.getctime(item_path)
            dirs.append((item_path, create_time))

    # 按创建时间降序排序
    dirs.sort(key=lambda x: x[1], reverse=True)

    # 删除超过max_keep的旧目录
    if len(dirs) > max_keep:
        for dir_path, _ in dirs[max_keep:]:
            try:
                shutil.rmtree(dir_path)
                print(f"🗑️  清理旧向量库目录：{dir_path}")
            except Exception as e:
                print(f"⚠️  清理旧目录失败 {dir_path}：{str(e)}")


# ===================== 财务RAG核心类 =====================
class FinanceRAG:
    """
    财务PDF问答RAG类（低显存适配，Qwen2 + BGE-small-zh）
    核心特性：
    1. 混合检索（向量检索为主 + 关键词检索为辅）
    2. 严格只回答问题，不扩展、不多答
    3. 兼容繁体财务术语
    4. 低显存分块策略
    5. 精准检索财务关键词
    6. 统一存储向量库目录，整洁易管理
    """

    def __init__(self):
        """初始化模型、嵌入和向量库"""
        # 初始化LLM（温度0.1保证回答稳定）
        self.llm: Optional[OllamaLLM] = OllamaLLM(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.1
        )

        # 初始化嵌入模型（轻量化）
        self.embeddings: Optional[OllamaEmbeddings] = OllamaEmbeddings(
            model=CONFIG["embed_model"],
            base_url=CONFIG["ollama_base_url"]
        )

        # 向量库初始化
        self.db: Optional[Chroma] = None
        self.current_db_path: Optional[str] = None  # 记录当前向量库路径

        # 关键词检索器（BM25）
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.split_docs = []  # 存储分块后的文档，供关键词检索使用

    def build_db(self, pdf_path: str) -> None:
        """
        构建PDF向量库 + 初始化关键词检索器（统一存储到父目录）
        :param pdf_path: PDF文件路径
        :raises Exception: 构建失败时抛出异常
        """
        # 1. 校验PDF文件存在性和格式
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF文件不存在：{pdf_path}")

        if not pdf_path.lower().endswith(".pdf"):
            raise ValueError(f"❌ 文件不是PDF格式：{pdf_path}")

        # 2. 创建统一的父目录（不存在则自动创建）
        os.makedirs(CONFIG["chroma_parent_dir"], exist_ok=True)

        # 3. 生成唯一向量库子目录（放在父目录下）
        unique_dir_name = f"chroma_db_{uuid.uuid4().hex[:8]}"
        self.current_db_path = os.path.join(CONFIG["chroma_parent_dir"], unique_dir_name)
        print(f"📌 本次向量库将存储到：{self.current_db_path}")

        # 4. 清理旧目录（保留最新5个，避免磁盘占满）
        clean_old_chroma_dirs(CONFIG["chroma_parent_dir"], max_keep=5)

        # 5. 加载PDF并清洗文本
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"📌 成功加载PDF，共 {len(docs)} 页")
        except Exception as e:
            raise Exception(f"❌ 加载PDF失败：{str(e)}")

        # 6. 文本清洗（过滤空内容/短文本）
        cleaned_docs = []
        for idx, doc in enumerate(docs):
            cleaned_text = clean_text(doc.page_content)
            # 过滤空文本或过短文本（长度<10）
            if cleaned_text and len(cleaned_text) > 10:
                cleaned_docs.append({
                    "page_content": cleaned_text,
                    "metadata": {"page": idx + 1}  # 页码从1开始
                })

        if not cleaned_docs:
            raise Exception("❌ PDF清洗后无有效文本内容")
        print(f"📌 PDF清洗完成，有效文本页数：{len(cleaned_docs)}")

        # 7. 文本分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]  # 中文优先分隔符
        )

        # 生成分块文档
        self.split_docs = splitter.create_documents(
            texts=[doc["page_content"] for doc in cleaned_docs],
            metadatas=[doc["metadata"] for doc in cleaned_docs]
        )
        print(f"📌 文本分块完成，有效分块数：{len(self.split_docs)}")

        # 8. 初始化关键词检索器（BM25）
        self.bm25_retriever = BM25Retriever.from_documents(self.split_docs)
        self.bm25_retriever.k = CONFIG["keyword_k"]  # 关键词检索返回数量
        print(f"📌 BM25关键词检索器初始化完成（返回{CONFIG['keyword_k']}条结果）")

        # 9. 构建向量库（存储到父目录下的唯一子目录）
        try:
            self.db = Chroma.from_documents(
                documents=self.split_docs,
                embedding=self.embeddings,
                persist_directory=self.current_db_path
            )
            print(f"✅ 向量库+关键词检索器构建完成，存储路径：{self.current_db_path}")
        except Exception as e:
            self.db = None
            self.current_db_path = None
            self.bm25_retriever = None
            raise Exception(f"❌ 构建向量库失败：{str(e)}")

    def hybrid_retrieval(self, query: str) -> List:
        """
        混合检索核心函数（向量为主 + 关键词为辅）
        :param query: 用户查询问题
        :return: 去重后的混合检索文档列表
        """
        # 1. 向量语义检索（主导）
        vector_docs_with_score = self.db.similarity_search_with_score(
            query,
            k=CONFIG["retrieve_k"] * 2  # 先取双倍数量，再过滤
        )
        # 过滤低相似度结果
        vector_docs = [doc for doc, score in vector_docs_with_score if score < CONFIG["similarity_threshold"]]
        # 兜底：如果过滤后无结果，取原始top_k
        if not vector_docs:
            vector_docs = [doc for doc, _ in vector_docs_with_score[:CONFIG["retrieve_k"]]]
        print(f"🔍 向量检索到 {len(vector_docs)} 条有效结果")

        # 2. 关键词精确检索（辅助）
        keyword_docs = self.bm25_retriever.get_relevant_documents(query)
        print(f"🔍 关键词检索到 {len(keyword_docs)} 条有效结果")

        # 3. 结果去重（按文本内容+页码去重）
        combined_docs = []
        seen = set()  # 记录已出现的文档特征（文本前100字符+页码）
        # 先加向量检索结果（保证主导地位）
        for doc in vector_docs:
            key = (doc.page_content[:100], doc.metadata.get("page", "unknown"))
            if key not in seen:
                seen.add(key)
                combined_docs.append(doc)
        # 再加关键词检索结果（补充未命中的）
        for doc in keyword_docs:
            key = (doc.page_content[:100], doc.metadata.get("page", "unknown"))
            if key not in seen:
                seen.add(key)
                combined_docs.append(doc)

        # 4. 控制最终结果数量
        final_docs = combined_docs[:CONFIG["retrieve_k"] + CONFIG["keyword_k"]]
        print(f"🔍 混合检索去重后最终结果数：{len(final_docs)}")

        return final_docs

    def query(self, question: str) -> str:
        """
        :param question: 用户问题
        :return: 格式化的回答（含页码）
        :raises Exception: 未构建向量库/查询失败时抛出异常
        """
        # 1. 校验向量库是否已构建
        if self.db is None or self.current_db_path is None or self.bm25_retriever is None:
            raise Exception("❌ 向量库/关键词检索器未构建，请先上传PDF并执行build_db()")

        # 2. 校验问题非空
        if not question or question.strip() == "":
            raise ValueError("❌ 查询问题不能为空")

        # 3. 关键词扩展（提升检索精准度）
        target_key = None
        expand_keys = []
        for key, aliases in FINANCE_KEYS.items():
            if key in question:
                target_key = key
                expand_keys = aliases
                break

        # 4. 构建扩展查询词
        expanded_query = question + " " + " ".join(expand_keys)

        # 5. 混合检索
        try:
            retrieved_docs = self.hybrid_retrieval(expanded_query)
        except Exception as e:
            raise Exception(f"❌ 混合检索失败：{str(e)}")

        # 6. 过滤仅包含目标关键词的文档（提升精准度）
        if target_key:
            filtered_docs = [doc for doc in retrieved_docs if target_key in doc.page_content]
            retrieved_docs = filtered_docs if filtered_docs else retrieved_docs

        # 7. 提取上下文和页码
        page_numbers = [str(doc.metadata.get("page", "未知")) for doc in retrieved_docs]
        context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])

        # 8. 构建提示词
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
规则（必须严格遵守）：
1. 只回答【{question}】这一个问题
2. 只从上下文找答案，不扩展、不联想、不额外提取信息
3. 只输出最终答案，不要指标名、不要解释、不要来源、不要格式
4. 找不到相关数据就输出：未找到相关数据

上下文：
{context}

只输出答案：
"""
        )

        # 9. 执行LLM推理
        try:
            chain = prompt_template | self.llm | StrOutputParser()
            raw_answer = chain.invoke({
                "context": context_text,
                "question": question
            }).strip()
        except Exception as e:
            raise Exception(f"❌ LLM生成回答失败：{str(e)}")

        # 10. 格式化回答（含数据来源页码）
        final_answer = f"{raw_answer}\n数据来源：第{','.join(page_numbers)}页"

        # 11. 打印检索日志（调试用）
        print(f"\n🔍 混合检索最终结果：")
        for doc in retrieved_docs:
            page = doc.metadata.get("page", "未知")
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(f"   第{page}页：{content_preview}")

        return final_answer


# ===================== 本地测试入口 =====================
if __name__ == "__main__":
    # 初始化RAG实例
    rag = FinanceRAG()

    # 构建向量库（替换为你的PDF路径）
    pdf_file_path = r"E:\conda\envs\llm_1_9\乐华年报.pdf"
    try:
        rag.build_db(pdf_file_path)
    except Exception as e:
        print(f"❌ 构建向量库失败：{str(e)}")
        exit(1)

    # 测试查询
    test_questions = [
        "首席执行官是？",
        "高级管理层有？",
        "营业收入是多少？",
        "净利润是多少？",
        "总资产是多少？",
        "总负债是多少？",
        "杜华女士在哪个学校毕业的？"
    ]

    print("\n" + "=" * 100)
    print("📊 财务PDF问答测试（混合检索）")
    print("=" * 100)
    for q in test_questions:
        print(f"\n📝 问题：{q}")
        try:
            answer = rag.query(q)
            print(f"✅ 回答：{answer}")
        except Exception as e:
            print(f"❌ 回答失败：{str(e)}")
    print("\n" + "=" * 100)