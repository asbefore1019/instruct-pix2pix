# 这段代码是一个完整的脚本，用于计算图像编辑模型的性能指标并生成相应的图表：
# 导入必要的库和模块。
# 定义了一个CFGDenoiser类，它是一个包装器，用于对模型进行配置，并在前向传播过程中应用相应的配置。
# 定义了一个load_model_from_config函数，用于从配置文件和检查点文件中加载模型。
# 定义了一个ImageEditor类，它是一个图像编辑器模型的封装类。它加载了配置文件和检查点文件，并提供了一个前向传播方法，用于对图像进行编辑。
# 定义了一个compute_metrics函数，用于计算模型的性能指标。它接受一些参数，如配置文件路径、模型路径、数据集路径等，并使用ImageEditor类和ClipSimilarity类来计算指标。
# 定义了一个plot_metrics函数，用于绘制性能指标的图表。
# 定义了一个main函数，它是脚本的主要入口点。它解析命令行参数，并调用compute_metrics和plot_metrics函数来计算指标并生成图表。
# 最后，通过调用main函数来执行整个脚本。


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
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

import json
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path

sys.path.append("./")

from clip_similarity import ClipSimilarity
from edit_dataset import EditDatasetEval

sys.path.append("./stable_diffusion")

from ldm.util import instantiate_from_config


# 定义了一个CFGDenoiser类，它是一个包装器，用于对模型进行配置，并在前向传播过程中应用相应的配置。
class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


# 定义了一个load_model_from_config函数，用于从配置文件和检查点文件中加载模型。
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

# 定义ImageEditor类，它是一个图像编辑器模型的封装类。它加载了配置文件和检查点文件，并提供了一个前向传播方法，用于对图像进行编辑。
class ImageEditor(nn.Module):
    def __init__(self, config, ckpt, vae_ckpt=None):
        super().__init__()
        
        config = OmegaConf.load(config)
        self.model = load_model_from_config(config, ckpt, vae_ckpt)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])
        # 在初始化方法中，该类加载了一个配置文件，并使用该配置文件创建了一个模型。模型的检查点文件路径由ckpt参数指定，可选的VAE模型的检查点文件路径由vae_ckpt参数指定。
        # 然后，将模型设置为评估模式，并将其移动到GPU上。接下来，创建了一个K.external.CompVisDenoiser对象，将模型包装在其中，以进行图像去噪。
        # 最后，初始化了一个特殊的标记null_token，用于表示一个空的条件。

    # forward方法用于执行图像编辑操作。它接受一个image参数，表示输入的图像。
    # edit参数是一个字符串，表示要对图像进行的编辑操作。scale_txt和scale_img参数分别用于控制文本和图像的缩放比例。steps参数表示采样步数。
    def forward(
        self,
        image: torch.Tensor,
        edit: str,
        scale_txt: float = 7.5,
        scale_img: float = 1.0,
        steps: int = 100,
    ) -> torch.Tensor:
        assert image.dim() == 3
        assert image.size(1) % 64 == 0
        assert image.size(2) % 64 == 0
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning([edit])],
                "c_concat": [self.model.encode_first_stage(image[None]).mode()],
            }
            uncond = {
                "c_crossattn": [self.model.get_learned_conditioning([""])],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])],
            }
            extra_args = {
                "uncond": uncond,
                "cond": cond,
                "image_cfg_scale": scale_img,
                "text_cfg_scale": scale_txt,
            }
            sigmas = self.model_wrap.get_sigmas(steps)
            x = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(x)[0]
            return x
"""在forward方法中，首先进行一些断言，确保输入的图像满足要求。然后，使用torch.no_grad()和autocast("cuda")上下文管理器，禁用梯度计算并将计算操作移动到GPU上。
接下来，根据编辑指令和图像，构建条件和非条件变量。然后，设置一些额外的参数，如无条件变量、条件变量、图像配置缩放比例和文本配置缩放比例。
接着，通过调用self.model_wrap.get_sigmas(steps)获取采样步数对应的标准差。然后，使用高斯分布采样生成一个与条件变量相同形状的随机张量x，并乘以第一个标准差。
接着，使用K.sampling.sample_euler_ancestral方法进行采样，得到修改后的图像。最后，将修改后的图像返回。"""


# 定义compute_metrics函数用于计算模型的性能指标。它接受参数，如配置文件路径、模型路径、数据集路径等，并使用ImageEditor类和ClipSimilarity类来计算指标
def compute_metrics(config,
                    model_path, 
                    vae_ckpt,
                    data_path,
                    output_path, 
                    scales_img, 
                    scales_txt, 
                    num_samples = 5000, 
                    split = "test", 
                    steps = 50, 
                    res = 512, 
                    seed = 0):
    editor = ImageEditor(config, model_path, vae_ckpt).cuda()
    clip_similarity = ClipSimilarity().cuda()



    outpath = Path(output_path, f"n={num_samples}_p={split}_s={steps}_r={res}_e={seed}.jsonl")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for scale_txt in scales_txt:
        for scale_img in scales_img:
            dataset = EditDatasetEval(
                    path=data_path, 
                    split=split, 
                    res=res
                    )
            assert num_samples <= len(dataset)
            print(f'Processing t={scale_txt}, i={scale_img}')
            torch.manual_seed(seed)
            perm = torch.randperm(len(dataset))
            count = 0
            i = 0

            sim_0_avg = 0
            sim_1_avg = 0
            sim_direction_avg = 0
            sim_image_avg = 0
            count = 0

            pbar = tqdm(total=num_samples)
            while count < num_samples:
                
                idx = perm[i].item()
                sample = dataset[idx]
                i += 1

                gen = editor(sample["image_0"].cuda(), sample["edit"], scale_txt=scale_txt, scale_img=scale_img, steps=steps)

                sim_0, sim_1, sim_direction, sim_image = clip_similarity(
                    sample["image_0"][None].cuda(), gen[None].cuda(), [sample["input_prompt"]], [sample["output_prompt"]]
                )
                sim_0_avg += sim_0.item()
                sim_1_avg += sim_1.item()
                sim_direction_avg += sim_direction.item()
                sim_image_avg += sim_image.item()
                count += 1
                pbar.update(count)
            pbar.close()

            sim_0_avg /= count
            sim_1_avg /= count
            sim_direction_avg /= count
            sim_image_avg /= count

            with open(outpath, "a") as f:
                f.write(f"{json.dumps(dict(sim_0=sim_0_avg, sim_1=sim_1_avg, sim_direction=sim_direction_avg, sim_image=sim_image_avg, num_samples=num_samples, split=split, scale_txt=scale_txt, scale_img=scale_img, steps=steps, res=res, seed=seed))}\n")
    return outpath

# 定义了一个plot_metrics函数，用于绘制性能指标的图表。
def plot_metrics(metrics_file, output_path):
    
    with open(metrics_file, 'r') as f:
        data = [json.loads(line) for line in f]
        
    plt.rcParams.update({'font.size': 11.5})
    seaborn.set_style("darkgrid")
    plt.figure(figsize=(20.5* 0.7, 10.8* 0.7), dpi=200)

    x = [d["sim_direction"] for d in data]
    y = [d["sim_image"] for d in data]

    plt.plot(x, y, marker='o', linewidth=2, markersize=4)

    plt.xlabel("CLIP Text-Image Direction Similarity", labelpad=10)
    plt.ylabel("CLIP Image Similarity", labelpad=10)

    plt.savefig(Path(output_path) / Path("plot.pdf"), bbox_inches="tight")

# main函数:解析命令行参数，并调用compute_metrics和plot_metrics函数来计算指标并生成图表
def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--output_path", default="analysis/", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--dataset", default="data/clip-filtered-dataset/", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    args = parser.parse_args()

    scales_img = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    scales_txt = [7.5]
    
    metrics_file = compute_metrics(
            args.config,
            args.ckpt, 
            args.vae_ckpt,
            args.dataset, 
            args.output_path, 
            scales_img, 
            scales_txt,
            steps = args.steps,
            )
    
    plot_metrics(metrics_file, args.output_path)
        


if __name__ == "__main__":
    main()
