import gradio as gr
import requests

# FastAPI接口地址
API_URL = "http://127.0.0.1:8000"


def upload_pdf(file):
    """上传PDF到FastAPI"""
    if not file:
        return "❌ 请上传PDF文件！"
    try:
        with open(file.name, "rb") as f:
            files = {"file": f}
            resp = requests.post(f"{API_URL}/upload-pdf", files=files)
        if resp.status_code == 200:
            return "✅ PDF上传并解析成功！"
        else:
            return f"❌ 上传失败：{resp.json()['detail']}"
    except Exception as e:
        return f"❌ 上传出错：{str(e)}"


def query_question(question):
    """调用FastAPI问答接口"""
    if not question or question.strip() == "":
        return "❌ 请输入问题！"
    try:
        data = {"question": question}
        resp = requests.post(f"{API_URL}/query", data=data)
        if resp.status_code == 200:
            return resp.json()["answer"]
        else:
            return f"❌ 查询失败：{resp.json()['detail']}"
    except Exception as e:
        return f"❌ 查询出错：{str(e)}"


# 创建Gradio界面
with gr.Blocks(title="财务PDF智能问答系统") as demo:
    gr.Markdown("# 📊 财务PDF智能问答系统\n基于Qwen + RAG（RTX3050）")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="上传财务PDF", file_types=[".pdf"])
            upload_btn = gr.Button("上传并解析", variant="primary")
            upload_status = gr.Textbox(label="上传状态", lines=1)

        with gr.Column(scale=2):
            question = gr.Textbox(label="输入问题", placeholder="例如：营业收入是多少？")
            query_btn = gr.Button("获取答案", variant="secondary")
            answer = gr.Textbox(label="回答结果", lines=8)

    # 绑定按钮事件
    upload_btn.click(upload_pdf, inputs=[pdf_file], outputs=[upload_status])
    query_btn.click(query_question, inputs=[question], outputs=[answer])

    gr.Markdown("### 📝 支持查询的指标：\n盈利情况、首席执行官、主要股东等")

# 启动Gradio
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=True,
        show_error=True,
        debug=True
    )