import os
import re
import shutil
import uuid
from typing import List, Dict, Optional
from opencc import OpenCC  # 导入专业繁简转换库

# LangChain 相关导入
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===================== 全局配置 =====================
CONFIG = {
    "llm_model": "qwen2",
    "embed_model": "quentinz/bge-small-zh-v1.5:latest",  # 轻量化嵌入模型
    "ollama_base_url": "http://localhost:11434",  # Ollama 服务地址
    "chunk_size":450,#文本分块大小（低显存适配）
    "chunk_overlap": 40,  # 分块重叠长度
    "retrieve_k": 3,  # 检索相似文本数量
    "chroma_parent_dir": "./chroma_db_parent",  # 统一的向量库父目录（核心优化）
}

# 财务关键词映射
FINANCE_KEYS = {
    "净利润": ["归母净利润", "归属于母公司净利润"],
    "营业收入": ["營收", "营业总收入"],
    "总资产": ["资产总额", "资产总计"],
    "总负债": ["负债总额", "负债总计"],
    "首席执行官": ["CEO", "行政总裁"],
    "艺人": ["签约艺人", "旗下艺人"],
    "高级管理层": ["管理層", "高管团队"]
}

# 初始化全局繁简转换器（t2s = 繁体转简体）
cc = OpenCC("t2s")

# ===================== 工具函数 =====================
def clean_text(text: str) -> str:
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

    # 核心替换：全自动繁简转换（删除手动映射表，改用专业库）
    try:
        text = cc.convert(text)  # 一键完成所有繁体→简体转换
    except Exception as e:
        print(f"⚠️  繁简转换异常：{str(e)}，使用原始文本")

    return text


def clean_old_chroma_dirs(parent_dir: str, max_keep: int = 5) -> None:
    """
    清理父目录下的旧向量库目录（可选功能）
    :param parent_dir: 向量库父目录
    :param max_keep: 保留最新的N个目录，其余删除
    """
    if not os.path.exists(parent_dir):
        return

    # 获取所有子目录，按创建时间排序（最新的在前）
    dirs = []
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            # 获取目录创建时间（Windows用ctime，Linux用stat）
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
    财务PDF问答RAG
    1. 严格只回答问题，不扩展、不多答
    2. 全自动繁简转换
    3. 低显存分块策略
    4. 精准检索财务关键词
    5. 统一存储向量库目录，整洁易管理
    """

    def __init__(self):
        """初始化模型、嵌入和向量库"""
        # 初始化LLM
        self.llm: Optional[OllamaLLM] = OllamaLLM(
            model=CONFIG["llm_model"],
            base_url=CONFIG["ollama_base_url"],
            temperature=0.1
        )

        # 初始化嵌入模型
        self.embeddings: Optional[OllamaEmbeddings] = OllamaEmbeddings(
            model=CONFIG["embed_model"],
            base_url=CONFIG["ollama_base_url"]
        )

        # 向量库初始化
        self.db: Optional[Chroma] = None
        self.current_db_path: Optional[str] = None  # 记录当前向量库路径

    def build_db(self, pdf_path: str) -> None:
        """
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

        # 4. 清理旧目录（保留最新5个）
        clean_old_chroma_dirs(CONFIG["chroma_parent_dir"], max_keep=5)

        # 5. 加载PDF并清洗文本
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"📌 成功加载PDF，共 {len(docs)} 页")
        except Exception as e:
            raise Exception(f"❌ 加载PDF失败：{str(e)}")

        # 6. 文本清洗
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
        print(f"📌 PDF清洗完成（含繁简转换），有效文本页数：{len(cleaned_docs)}")

        # 7. 文本分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", "。", "，", "：", " "]  # 中文优先分隔符
        )

        # 生成分块文档
        split_docs = splitter.create_documents(
            texts=[doc["page_content"] for doc in cleaned_docs],
            metadatas=[doc["metadata"] for doc in cleaned_docs]
        )
        print(f"📌 文本分块完成，有效分块数：{len(split_docs)}")

        # 8. 构建向量库（存储到唯一子目录）
        try:
            self.db = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.current_db_path
            )
            print(f"✅ 向量库构建完成，存储路径：{self.current_db_path}")
        except Exception as e:
            self.db = None
            self.current_db_path = None
            raise Exception(f"❌ 构建向量库失败：{str(e)}")

    def query(self, question: str) -> str:
        """
        """
        # 1. 校验向量库是否已构建
        if self.db is None or self.current_db_path is None:
            raise Exception("❌ 向量库未构建，请先上传PDF并执行build_db()")

        # 2. 校验问题非空
        if not question or question.strip() == "":
            raise ValueError("❌ 查询问题不能为空")

        # 3. 关键词扩展（提升检索精准度，扩展词也转简体）
        target_key = None
        expand_keys = []
        for key, aliases in FINANCE_KEYS.items():
            if key in question:
                target_key = key
                # 扩展词转为简体，保证和财报文本格式一致
                expand_keys = [cc.convert(alias) for alias in aliases]
                break

        # 4. 构建扩展查询词（用户问题也转简体）
        expanded_query = cc.convert(question) + " " + " ".join(expand_keys)

        # 5. 相似性检索（增加相似度得分过滤）
        try:
            # 先检索更多结果，再过滤低相似度的
            retrieved_docs_with_score = self.db.similarity_search_with_score(
                expanded_query,  # 用简体查询词检索
                k=CONFIG["retrieve_k"] * 2  # 取双倍数量候选
            )
            # 过滤相似度得分<0.3的结果（得分越低越相似，可根据实际情况调整）
            retrieved_docs = [doc for doc, score in retrieved_docs_with_score if score < 0.3]
            # 兜底：如果过滤后无结果，取原始top3
            if not retrieved_docs:
                retrieved_docs = [doc for doc, _ in retrieved_docs_with_score[:CONFIG["retrieve_k"]]]
        except Exception as e:
            raise Exception(f"❌ 检索相似文本失败：{str(e)}")

        # 6. 过滤仅包含目标关键词的文档（提升精准度，关键词转简体）
        if target_key:
            target_key = cc.convert(target_key)  # 转为简体后再过滤
            filtered_docs = [doc for doc in retrieved_docs if target_key in doc.page_content]
            retrieved_docs = filtered_docs if filtered_docs else retrieved_docs

        # 7. 提取上下文和页码
        page_numbers = [str(doc.metadata.get("page", "未知")) for doc in retrieved_docs]
        context_text = "\n---\n".join([doc.page_content for doc in retrieved_docs])

        # 8. 构建严格的提示词（确保只回答问题）
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
                "question": question  # 保留原始问题（用户可能输入繁体/简体）
            }).strip()
        except Exception as e:
            raise Exception(f"❌ LLM生成回答失败：{str(e)}")

        # 10. 格式化回答（含数据来源页码）
        final_answer = f"{raw_answer}\n数据来源：第{','.join(page_numbers)}页"

        # 11. 打印检索日志（调试用）
        print(f"\n🔍 检索到{len(retrieved_docs)}条相关上下文：")
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
    print("📊 财务PDF问答测试（全自动繁简转换版）")
    print("=" * 100)
    for q in test_questions:
        print(f"\n📝 问题：{q}")
        try:
            answer = rag.query(q)
            print(f"✅ 回答：{answer}")
        except Exception as e:
            print(f"❌ 回答失败：{str(e)}")
    print("\n" + "=" * 100)