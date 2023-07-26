from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


# `CFGDenoiser`的PyTorch模型类接受一个模型作为参数，并在前向传播过程中对输入进行处理。
# 在`forward`方法中，输入包括`z`（噪声向量）、`sigma`（噪声标准差）、`cond`（条件输入）和`uncond`（无条件输入）。`text_cfg_scale`和`image_cfg_scale`是两个缩放因子。
# 在前向传播过程中，首先使用`einops.repeat`函数将`z`和`sigma`在第一个维度上重复3次，以匹配`cfg_cond`中的维度。然后，将`cond`和`uncond`中的特征拼接起来，并使用`torch.cat`函数将它们连接在一起。
# 接下来，将重复后的`z`、`sigma`和`cfg_cond`作为参数传递给内部模型`self.inner_model`进行前向传播。前向传播的结果被分成3部分，分别为`out_cond`、`out_img_cond`和`out_uncond`。
# 最后，根据给定的缩放因子，计算输出结果并返回。输出结果由`out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)`给出。

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    # einops.repeat函数可以在指定的维度上重复张量的元素;einops.repeat(tensor, pattern)，
    # 其中tensor是要重复的张量，pattern是一个字符串，用于指定重复的模式。模式字符串中的每个字符对应于tensor的一个维度，可以使用数字或字母来表示维度的大小。
    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

# 这段代码是一个函数`load_model_from_config`，根据给定的配置和检查点文件加载模型。

# 函数的参数包括：
# - `config`：模型的配置信息。
# - `ckpt`：模型的检查点文件路径。
# - `vae_ckpt`：VAE（变分自动编码器）的检查点文件路径，默认为None。
# - `verbose`：是否打印详细信息，默认为False。

# 函数的主要步骤如下：
# 1. 打印加载模型的信息。
# 2. 使用`torch.load`函数加载检查点文件，将结果存储在`pl_sd`变量中，并使用`map_location="cpu"`将模型加载到CPU上。
# 3. 如果`pl_sd`中包含"global_step"键，打印全局步骤信息。
# 4. 从`pl_sd`中提取"state_dict"，存储在`sd`变量中。
# 5. 如果`vae_ckpt`不为None，打印加载VAE模型的信息。
# 6. 使用`torch.load`函数加载VAE检查点文件，将结果存储在`vae_sd`变量中，并使用`map_location="cpu"`将模型加载到CPU上。
# 7. 根据模型的命名规则，将`sd`中的键进行转换，将"first_stage_model."开头的键转换为对应在VAE模型中的键。
# 8. 使用`instantiate_from_config`函数根据配置信息实例化模型，得到模型对象。
# 9. 使用模型对象的`load_state_dict`方法加载`sd`的状态字典，设置`strict=False`允许部分键不匹配。
# 10. 如果有缺失的键（m）或意外的键（u），并且`verbose`为True，则打印缺失和意外的键。
# 11. 返回加载的模型对象。

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


# 主函数用于生成编辑后的图像：
# 1. 首先，通过解析命令行参数，获取输入和输出文件路径、配置文件路径、检查点文件路径等信息。
# 2. 使用OmegaConf库加载配置文件。
# 3. 使用load_model_from_config函数加载模型，该函数根据配置文件和检查点文件创建模型。
# 4. 将模型设置为评估模式，并将其移动到GPU上。
# 5. 创建模型的包装器，用于进行图像处理。
# 6. 创建一个空的条件变量token。
# 7. 如果没有指定随机种子，则生成一个随机种子。
# 8. 打开输入图像，并将其转换为RGB格式。
# 9. 根据给定的分辨率调整图像的大小。
# 10. 如果没有指定编辑内容，则直接保存输入图像并返回。
# 11. 使用torch.no_grad()和autocast("cuda")上下文管理器，禁用梯度计算，并将模型切换到EMA模式。
# 12. 创建条件变量cond，将编辑内容编码为条件向量。
# 13. 创建无条件变量uncond，用于生成图像的其他部分。
# 14. 获取模型包装器的标准差。
# 15. 设置额外的参数。
# 16. 使用随机种子生成初始隐变量z。
# 17. 使用模型包装器和额外的参数进行采样，生成编辑后的图像。
# 18. 对图像进行后处理，将像素值限制在0到255之间，并转换为整数类型。
# 19. 保存编辑后的图像。

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    input_image = Image.open(args.input).convert("RGB")
    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        input_image.save(args.output)
        return

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(args.steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()

    
# 这段代码是一个图像去噪的脚本。它使用了一个预训练的模型来对输入图像进行去噪处理。代码的主要部分：
# 1. 导入必要的库和模块。
# 2. 定义了一个名为`CFGDenoiser`的类，它继承自`nn.Module`。这个类包装了一个给定的模型，用于图像去噪。
# 3. 定义了一个名为`load_model_from_config`的函数，用于从配置文件中加载模型。
# 4. 定义了一个名为`main`的函数，它是脚本的主要逻辑。在这个函数中，首先解析命令行参数，然后加载模型和输入图像。接下来，根据编辑指令和配置参数生成条件向量，并使用模型对输入图像进行去噪处理。最后，将处理后的图像保存到输出文件中。
# 5. 在`__main__`块中调用`main`函数。
