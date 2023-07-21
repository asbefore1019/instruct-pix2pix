from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import openai
from tqdm.auto import tqdm


DELIMITER_0 = "\n##\n"
DELIMITER_1 = "\n%%\n"
STOP = "\nEND"


def generate(
    openai_model: str,
    caption: str,
    num_retries: int = 3,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.1,
    presence_penalty: float = 0.0,
    sleep_on_error: float = 1.0,
) -> Optional[tuple[str, str]]:
    for _ in range(1 + num_retries):
        try:
            response = openai.Completion.create(
                model=openai_model,
                prompt=caption + DELIMITER_0,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=[STOP],
            )
        except Exception as e:
            print(e)
            time.sleep(sleep_on_error)
            continue
        output = response["choices"][0]["text"].split(DELIMITER_1)
        if len(output) == 2:
            instruction, edited_caption = output
            results = openai.Moderation.create([instruction, edited_caption])["results"]
            if results[0]["flagged"] or results[1]["flagged"]:
                continue
            if caption.strip().strip(".!?").lower() != edited_caption.strip().strip(".!?").lower():
                return instruction, edited_caption


def main(openai_model: str, num_samples: int, num_partitions: int, partition: int, seed: int):
    dataset = datasets.load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus", split="train")
    # Other datasets we considered that may be worth trying:
    # dataset = datasets.load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train")
    # dataset = datasets.load_dataset("laion/laion-coco", split="train")

    np.random.seed(seed)
    permutation = np.array_split(np.random.permutation(len(dataset)), num_partitions)[partition]
    dataset = dataset[permutation]
    captions = dataset["TEXT"]
    urls = dataset["URL"]
    output_path = f"data/dataset=laion-aesthetics-6.5_model={openai_model}_samples={num_samples}_partition={partition}.jsonl"  # fmt: skip
    print(f"Prompt file path: {output_path}")

    count = 0
    caption_set = set()
    url_set = set()

    if Path(output_path).exists():
        with open(output_path, "r") as f:
            for line in tqdm(f, desc="Resuming from existing prompts"):
                prompt = json.loads(line)
                if prompt["caption"] not in caption_set and prompt["url"] not in url_set:
                    caption_set.add(prompt["caption"])
                    url_set.add(prompt["url"])
                    count += 1

    with open(output_path, "a") as fp:
        with tqdm(total=num_samples - count, desc="Generating instructions and edited captions") as progress_bar:
            for caption, url in zip(captions, urls):
                if caption in caption_set or url in url_set:
                    continue
                if openai.Moderation.create(caption)["results"][0]["flagged"]:
                    continue
                edit_output = generate(openai_model, caption)
                if edit_output is not None:
                    edit, output = edit_output
                    fp.write(f"{json.dumps(dict(caption=caption, edit=edit, output=output, url=url))}\n")
                    count += 1
                    progress_bar.update()
                    caption_set.add(caption)
                    url_set.add(url)
                if count == num_samples:
                    break


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--openai-api-key", required=True, type=str)
    parser.add_argument("--openai-model", required=True, type=str)
    parser.add_argument("--num-samples", default=10000, type=int)
    parser.add_argument("--num-partitions", default=1, type=int)
    parser.add_argument("--partition", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    openai.api_key = args.openai_api_key
    main(args.openai_model, args.num_samples, args.num_partitions, args.partition, args.seed)


# 这段代码是一个用于生成样本的主要循环代码。它使用OpenAI的GPT-3模型生成编辑后的标题。下面是代码的主要步骤：

# 1. 导入所需的库和模块。
# 2. 定义了一些常量，如分隔符和停止标记。
# 3. 定义了一个`generate`函数，用于生成编辑后的标题。该函数使用OpenAI的Completion API来生成文本。
# 4. 定义了一个`main`函数，用于执行主要的样本生成逻辑。
# 5. 加载数据集，选择要使用的数据集。
# 6. 设置随机种子，并根据给定的分区数和分区索引，将数据集划分为多个分区。
# 7. 如果之前已经生成了一部分样本，从保存的文件中读取已生成的样本，并跳过这些样本。
# 8. 循环遍历数据集中的每个标题和URL。
# 9. 检查标题和URL是否已经在之前的样本中出现过，如果是，则跳过。
# 10. 使用OpenAI的Moderation API检查标题是否被标记为不合适的内容，如果是，则跳过。
# 11. 使用`generate`函数生成编辑后的标题。
# 12. 如果成功生成了编辑后的标题，则将原始标题、编辑说明和编辑后的标题以JSON格式写入输出文件。
# 13. 更新计数器和进度条。
# 14. 如果达到了指定的样本数量，停止生成样本。
# 15. 解析命令行参数，并调用`main`函数开始生成样本。
