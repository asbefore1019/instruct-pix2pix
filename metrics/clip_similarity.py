from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features
# `encode_text`方法用于将文本编码为特征向量。该方法详细说明：
# 1. `text`参数是一个包含多个文本的列表。
# 2. 使用`clip.tokenize`函数将文本列表转换为CLIP模型可接受的输入格式。`truncate=True`表示如果文本长度超过模型的最大限制，则进行截断。
# 3. 将文本转换为与模型参数所在设备相匹配的张量。
# 4. 使用CLIP模型的`encode_text`方法将文本转换为特征向量。
# 5. 对特征向量进行归一化，即将其除以其范数（欧几里德范数）。
# 6. 返回归一化后的文本特征向量。
# 请注意，这里使用了`self.model.encode_text`方法，该方法是CLIP模型的一部分，用于将文本编码为特征向量。

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
# encode_image方法用于将图像编码为特征向量：
# 首先，输入的图像被插值到指定的大小。这里使用的是双三次插值方法，将图像的尺寸调整为self.size。
# 接下来，从图像中减去平均值并除以标准差。这是为了对图像进行归一化处理，以便与CLIP模型的训练数据具有相似的分布。
# 然后，使用CLIP模型对图像进行编码，得到图像的特征向量。
# 最后，对特征向量进行归一化处理，使其具有单位长度。这是通过将特征向量除以其自身的范数（即向量的长度）来实现的。
# 最终，该方法返回图像的特征向量作为结果。
    
    def forward(
        self, image_0: torch.Tensor, image_1: torch.Tensor, text_0: list[str], text_1: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image

# 整段代码定义了一个名为`ClipSimilarity`的模型类，用于计算图像和文本之间的相似度。该类使用OpenAI的CLIP模型进行特征提取，并计算图像和文本之间的余弦相似度。

# 代码的主要部分：
# 1. 导入所需的库和模块。
# 2. 定义了一个`ClipSimilarity`模型类，继承自`nn.Module`。
# 3. 在`__init__`方法中，加载指定的CLIP模型，并设置模型为评估模式，不需要梯度。
# 4. 注册了用于归一化图像的均值和标准差。
# 5. 定义了`encode_text`方法，用于将文本编码为特征向量。
# 6. 定义了`encode_image`方法，用于将图像编码为特征向量。
# 7. 定义了`forward`方法，用于计算图像和文本之间的相似度。
# 8. 在`forward`方法中，首先将输入的图像和文本分别编码为特征向量。然后使用余弦相似度计算图像和文本之间的相似度。
#    返回四个相似度值：`sim_0`表示图像0和文本0之间的相似度，`sim_1`表示图像1和文本1之间的相似度，
#    `sim_direction`表示图像和文本之间的方向相似度，`sim_image`表示图像0和图像1之间的相似度。
