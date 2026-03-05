from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile

# 导入核心RAG类，在这里更换对应tool即可使用不同的模型or工具
from coretool import FinanceRAG, CONFIG

# 全局初始化一个RAG实例
rag = FinanceRAG()

app = FastAPI(
    title="财务文档RAG问答API",
    description="基于Qwen2的财务PDF问答接口",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = tempfile.mkdtemp()


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # 1. 校验文件类型
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持PDF文件")

        # 2. 保存临时文件
        pdf_path = os.path.join(TEMP_DIR, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # 3. 调用核心方法构建向量库
        rag.build_db(pdf_path)

        # 4. 删除临时文件
        os.remove(pdf_path)

        # 5. 校验向量库是否真的构建成功
        if rag.db is None:
            raise HTTPException(status_code=500, detail="向量库构建失败，请检查PDF文件或日志")

        return {"status": "success", "message": "✅ PDF解析完成，向量库构建成功"}

    except Exception as e:
        # 构建失败时重置向量库，避免后续查询报错
        rag.db = None
        raise HTTPException(status_code=500, detail=f"❌ 处理失败：{str(e)}")


@app.post("/query")
async def query(question: str = Form(...)):
    try:
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 先校验向量库是否存在
        if rag.db is None:
            raise HTTPException(status_code=400, detail="⚠️ 请先上传PDF并点击「上传并解析」按钮！")

        result = rag.query(question)
        return {"status": "success", "answer": result}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ 查询失败：{str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_fastapi:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )