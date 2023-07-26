from __future__ import annotations

from argparse import ArgumentParser

import datasets
import gradio as gr
import numpy as np
import openai

from dataset_creation.generate_txt_dataset import generate


def main(openai_model: str):
    dataset = datasets.load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus", split="train")
    captions = dataset[np.random.permutation(len(dataset))]["TEXT"]
    index = 0

    # 在数据集中随机选择一个图像描述
    def click_random():
        nonlocal index
        output = captions[index]
        index = (index + 1) % len(captions)
        return output

    # 接受一个输入图像描述，并使用generate函数生成编辑指令
    def click_generate(input: str):
        if input == "":
            raise gr.Error("Input caption is missing!")
        edit_output = generate(openai_model, input)
        if edit_output is None:
            return "Failed :(", "Failed :("
        return edit_output

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        txt_input = gr.Textbox(lines=3, label="Input Caption", interactive=True, placeholder="Type image caption here...")  # fmt: skip
        txt_edit = gr.Textbox(lines=1, label="GPT-3 Instruction", interactive=False)
        txt_output = gr.Textbox(lines=3, label="GPT3 Edited Caption", interactive=False)

        with gr.Row():
            clear_btn = gr.Button("Clear")
            random_btn = gr.Button("Random Input")
            generate_btn = gr.Button("Generate Instruction + Edited Caption")

            clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[txt_input, txt_edit, txt_output])
            random_btn.click(fn=click_random, inputs=[], outputs=[txt_input])
            generate_btn.click(fn=click_generate, inputs=[txt_input], outputs=[txt_edit, txt_output])

    demo.launch(share=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--openai-api-key", required=True, type=str)
    parser.add_argument("--openai-model", required=True, type=str)
    args = parser.parse_args()
    openai.api_key = args.openai_api_key
    main(args.openai_model)


# 这是一个使用gradio库构建的交互式应用程序，用于编辑图像描述。应用程序使用OpenAI的GPT-3模型生成编辑指令，并将其应用于给定的图像描述。下面是主要的代码逻辑：
#     导入所需的库和模块。
#     定义main函数，它接受一个openai_model参数。
#     加载数据集，这里使用了名为ChristophSchuhmann/improved_aesthetics_6.5plus的数据集。
#     定义click_random函数，用于在数据集中随机选择一个图像描述。
#     定义click_generate函数，它接受一个输入图像描述，并使用generate函数生成编辑指令。
#     使用gradio库创建一个交互式应用程序。
#     定义输入和输出文本框。
#     定义清空、随机选择和生成指令的按钮，并分别指定它们的点击事件。
#     解析命令行参数，包括openai-api-key和openai-model。
#     设置OpenAI的API密钥。
#     调用main函数，并传入openai-model参数。
# 请注意，这个脚本中使用的generate函数是从dataset_creation.generate_txt_dataset模块中导入的
