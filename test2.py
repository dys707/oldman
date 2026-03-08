import gradio as gr

def echo(message):
    return f"您说的是：{message}"

with gr.Blocks() as demo:
    inp = gr.Textbox(label="输入")
    out = gr.Textbox(label="输出")
    btn = gr.Button("提交")
    btn.click(fn=echo, inputs=inp, outputs=out)

demo.launch(server_name="127.0.0.1", server_port=7861, debug=True)