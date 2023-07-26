from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


# EditDataset类用于训练集，它继承自torch.utils.data.Dataset类。在初始化时，它接收一些参数，包括数据集路径、划分方式、尺寸调整参数等。它会根据划分方式从数据集中选择相应的样本。
# 在__getitem__方法中，它会读取相应的图像文件和prompt文件，并进行一系列的预处理操作，包括调整图像尺寸、归一化、随机裁剪和水平翻转等。最后返回一个字典，包含编辑后的图像和编辑相关的信息。

class EditDataset(Dataset):
    
    # path：数据集的路径。
    # split：数据集的划分，可以是"train"、"val"或"test"。
    # splits：一个包含三个浮点数的元组，表示训练集、验证集和测试集的划分比例。三个比例相加应等于1。
    # min_resize_res：图像调整尺寸的最小值。
    # max_resize_res：图像调整尺寸的最大值。
    # crop_res：图像裁剪的尺寸。
    # flip_prob：图像水平翻转的概率。

    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    # __len__方法返回数据集的长度，即数据样本的数量
    def __len__(self) -> int:
        return len(self.seeds)

    # __getitem__方法根据索引获取单个样本。它首先根据索引获取样本的名称和种子信息，然后加载图像和对应的prompt。
    # 接下来，对图像进行尺寸调整、归一化和随机裁剪等预处理操作。最后，返回一个包含编辑图像和编辑信息的字典。

"""首先，根据索引i获取样本的名称和种子。然后，根据名称构建样本的路径propt_dir。接下来，从种子列表中随机选择一个种子。
然后，打开样本路径下的prompt.json文件，并加载其中的"edit"字段作为prompt。
接下来，打开两个图片文件{seed}_0.jpg和{seed}_1.jpg，并进行图像尺寸调整操作。调整后的尺寸由self.min_resize_res和self.max_resize_res之间的一个随机整数确定。调整使用的方法是Image.Resampling.LANCZOS。
然后，将图像转换为张量，并进行归一化操作。归一化范围是[-1, 1]，并使用rearrange函数将通道维度移到最前面。
接下来，进行随机裁剪和水平翻转操作。随机裁剪使用RandomCrop函数，裁剪尺寸为self.crop_res。水平翻转的概率由self.flip_prob确定。最后，将裁剪和翻转后的图像切分成两个部分，分别命名为image_0和image_1。
最后，将处理后的图像和相关信息封装成字典返回。字典的键为edited和edit，对应的值分别为处理后的图像image_1和一个包含image_0和prompt的字典。"""
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)["edit"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))
        image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.jpg"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)  # 随机裁剪
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)  # 水平翻转

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))

# EditDatasetEval类用于评估集，它的功能与EditDataset类类似，但没有进行随机裁剪和翻转操作。

class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res
    # path: 数据集的路径
    # split: 数据集的划分方式，可以是"train"、"val"或"test"
    # splits: 数据集划分的比例，默认为(0.9, 0.05, 0.05)，表示训练集、验证集和测试集的比例
    # res: 图像的尺寸，默认为256

        # 从路径中加载"seeds.json"文件，该文件包含了图像的种子数据。将种子数据保存在self.seeds中
        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        # 根据split参数确定当前数据集划分的起始和结束位置。根据起始和结束位置，从self.seeds中选择对应的种子数据
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    # __len__方法返回数据集的长度，即self.seeds的长度
    def __len__(self) -> int:
        return len(self.seeds)

    """__getitem__方法用于获取数据集中的一个样本。首先根据索引i获取对应的种子数据和路径信息。
    然后读取对应的prompt.json文件，获取输入提示、编辑内容和输出提示。
    接着，打开图像文件并进行尺寸调整，调整后的尺寸由res参数指定。
    最后，将图像转换为张量并返回包含图像、输入提示、编辑内容和输出提示的字典。"""
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)
