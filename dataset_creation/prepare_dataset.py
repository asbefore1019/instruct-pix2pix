import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm.auto import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    seeds = []
    with tqdm(desc="Listing dataset image seeds") as progress_bar:
        for prompt_dir in dataset_dir.iterdir():
            if prompt_dir.is_dir():
                prompt_seeds = [image_path.name.split("_")[0] for image_path in sorted(prompt_dir.glob("*_0.jpg"))]
                if len(prompt_seeds) > 0:
                    seeds.append((prompt_dir.name, prompt_seeds))
                    progress_bar.update()
    seeds.sort()

    with open(dataset_dir.joinpath("seeds.json"), "w") as f:
        json.dump(seeds, f)


if __name__ == "__main__":
    main()


# 这段代码是一个Python脚本，用于处理文本数据集。它读取一个包含JSON数据的输入文件夹，并对每个JSON对象进行处理，然后将处理结果写入到一个输出文件中。
# 代码使用了`argparse`模块来解析命令行参数，其中唯一的参数是`dataset_dir`，表示输入数据集所在的文件夹路径。

# 首先，代码通过`Path`类将输入的文件夹路径转换为一个`Path`对象，并将其赋值给`dataset_dir`变量。
# 接下来，代码创建了一个空列表`seeds`，用于存储处理结果。
# 然后，代码使用`tqdm`模块创建了一个进度条，用于显示处理进度。通过遍历输入文件夹中的每个子文件夹，代码判断子文件夹是否存在，并且文件夹中是否存在以`_0.jpg`结尾的文件。
# 如果满足条件，代码提取文件名中的种子值，并将其添加到`seeds`列表中。在遍历过程中，进度条会更新。
# 最后，代码对`seeds`列表进行排序，并将结果写入到一个名为`seeds.json`的输出文件中。

# 整个处理过程在`main`函数中完成，然后通过`__name__`变量判断是否运行为主程序，如果是，则调用`main`函数来执行处理。
