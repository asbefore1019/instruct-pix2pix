import json
from argparse import ArgumentParser

from generate_txt_dataset import DELIMITER_0, DELIMITER_1, STOP


def main(input_path: str, output_path: str):
    with open(input_path) as f:
        prompts = [json.loads(l) for l in f]

    with open(output_path, "w") as f:
        for prompt in prompts:
            prompt_for_gpt = {
                "prompt": f"{prompt['input']}{DELIMITER_0}",
                "completion": f"{prompt['edit']}{DELIMITER_1}{prompt['output']}{STOP}",
            }
            f.write(f"{json.dumps(prompt_for_gpt)}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    main(args.input_path, args.output_path)


# 这段代码用于处理文本数据集：

# 1. 首先，代码导入了`json`模块和`ArgumentParser`类。
# 2. 接下来，代码从`generate_txt_dataset`模块中导入了一些常量，包括`DELIMITER_0`、`DELIMITER_1`和`STOP`。
# 3. 然后，定义了一个名为`main`的函数，该函数接受两个参数：`input_path`和`output_path`，用于指定输入文件和输出文件的路径。
# 4. 在`main`函数中，代码使用`with open(input_path) as f`打开输入文件，并将文件中的每一行解析为JSON格式的数据，存储在`prompts`列表中。
# 5. 然后，代码使用`with open(output_path, "w") as f`打开输出文件，并对`prompts`列表中的每个元素进行处理。
# 6. 对于每个`prompt`，代码构建了一个`prompt_for_gpt`字典，其中包含两个键值对：
#    - `"prompt"`键的值是一个字符串，由`prompt`字典中的`input`键的值和`DELIMITER_0`常量组成。
#    - `"completion"`键的值是一个字符串，由`prompt`字典中的`edit`键的值、`DELIMITER_1`常量、`prompt`字典中的`output`键的值和`STOP`常量组成。
# 7. 最后，代码使用`json.dumps`将`prompt_for_gpt`字典转换为JSON格式的字符串，并将其写入输出文件中。

# 在代码的最后部分，使用`ArgumentParser`类解析命令行参数，并调用`main`函数，将解析得到的参数传递给`main`函数进行处理。

# 总体来说，这段代码的功能是读取一个输入文件中的JSON数据，对每个JSON对象进行处理，然后将处理结果写入到一个输出文件中。
# 处理的过程是将`input`和`edit`字段拼接起来作为GPT的输入，将`edit`和`output`字段拼接起来作为GPT的输出。
