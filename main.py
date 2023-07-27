import argparse, os, sys, datetime, glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import json
import pickle

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image

import torch.distributed as dist
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.plugins import DDPPlugin

sys.path.append("./stable_diffusion")

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


# 创建一个参数解析器对象。该解析器用于解析命令行参数，并将其转换为相应的Python对象。在这个函数中，我们定义了一个内部函数str2bool，用于将字符串转换为布尔值。

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
# 创建了一个argparse.ArgumentParser对象，并使用add_argument方法添加了一个名为--name（或-n）的参数。
# 该参数接受一个字符串类型的值，并具有一些额外的选项，如const、default和nargs。这些选项用于指定参数的默认值、接受的值的数量以及帮助文档中的描述。
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
# 这段代码定义了一个函数nondefault_trainer_args(opt)，它接受一个opt参数，并返回一个列表，其中包含与默认训练器参数不同的参数名称。
# 函数首先创建一个argparse.ArgumentParser对象，并将其用于Trainer.add_argparse_args方法。
# 然后，它创建一个空的args对象，使用parser.parse_args([])方法解析命令行参数并将结果赋值给args。
# 接下来，函数使用列表推导式来生成一个包含与opt对象中的属性值不同的属性名称的列表。这是通过比较opt对象和args对象中的相应属性值来完成的。
# 最后，函数返回按字母顺序排序的属性名称列表。


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset
    将任意对象包装成一个PyTorch数据集，以便在训练和评估模型时使用"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    # 返回self.data中对应索引的样本
    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)
        
"""这段代码是一个用于初始化PyTorch数据加载器的函数`worker_init_fn`。它在每个工作进程中被调用，用于设置每个工作进程的随机种子和数据集的划分。
函数首先获取当前工作进程的信息，包括数据集和工作进程的ID。然后，它检查数据集是否是`Txt2ImgIterableBaseDataset`的一个实例。如果是，它将根据工作进程的ID划分数据集，并设置随机种子以确保每个工作进程使用不同的随机数序列。这是为了确保每个工作进程在处理数据时使用不同的随机数。
如果数据集不是`Txt2ImgIterableBaseDataset`的实例，它将使用相同的随机种子设置所有工作进程的随机数序列。
这个函数的目的是确保在多进程数据加载过程中，每个工作进程使用不同的随机数序列，以增加数据的随机性和多样性。"""


# 这是一个继承自`pl.LightningDataModule`的数据模块类`DataModuleFromConfig`。它用于配置和管理训练、验证、测试和预测数据集的加载器。

# 在`__init__`方法中，它接受一些参数，包括`batch_size`（批量大小），`train`、`validation`、`test`和`predict`（数据集配置），`wrap`（是否使用`WrappedDataset`包装数据集），`num_workers`（工作进程数），`shuffle_test_loader`（是否在测试数据加载器中打乱数据顺序），`use_worker_init_fn`（是否使用`worker_init_fn`函数初始化工作进程），`shuffle_val_dataloader`（是否在验证数据加载器中打乱数据顺序）。
# `prepare_data`方法用于准备数据，它遍历数据集配置并调用`instantiate_from_config`函数来实例化数据集。
# `setup`方法用于设置数据模块，它根据数据集配置实例化数据集，并根据`wrap`参数决定是否将数据集包装成`WrappedDataset`。
# `_train_dataloader`、`_val_dataloader`、`_test_dataloader`和`_predict_dataloader`方法分别返回训练、验证、测试和预测数据加载器。这些加载器使用`DataLoader`来加载数据集，并根据参数设置进行配置，如批量大小、工作进程数、是否打乱数据顺序等。
# 总的来说，`DataModuleFromConfig`类提供了一个方便的方式来配置和管理数据加载器，使得训练、验证、测试和预测数据集的加载变得更加简单和灵活。

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, persistent_workers=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=True)


# 这是一个名为`SetupCallback`的自定义回调类，它继承自`Callback`类。这个回调类在训练过程中的不同阶段执行一些操作。
# 在`__init__`方法中，该类接收一些参数，包括`resume`、`now`、`logdir`、`ckptdir`、`cfgdir`、`config`和`lightning_config`等。
# 在`on_keyboard_interrupt`方法中，如果`trainer.global_rank`为0，它会打印一条消息并保存最后一个检查点。检查点的路径是根据`self.ckptdir`和"last.ckpt"拼接而成的。
# 在`on_pretrain_routine_start`方法中，如果`trainer.global_rank`为0，它会创建日志目录和配置文件目录。如果`self.lightning_config`中包含"callbacks"并且其中包含"metrics_over_trainsteps_checkpoint"，它会创建一个名为"trainstep_checkpoints"的目录。然后，它会打印项目配置和Lightning配置，并将它们保存到相应的配置文件中。
# 这个回调类的作用是在训练过程中执行一些设置和保存操作，以便在训练过程中出现中断时能够恢复训练。

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            # os.makedirs(self.logdir, exist_ok=True)
            # os.makedirs(self.ckptdir, exist_ok=True)
            # os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

# 这段代码实现了一个名为`all_gather`的函数，用于在分布式训练中收集来自各个进程的数据。它接受一个可pickle的对象作为输入，并返回一个列表，其中包含从每个进程中收集到的数据。

# 函数的实现逻辑如下：
# 1. 首先，通过调用`get_world_size()`函数获取当前的进程数量，如果只有一个进程，则直接返回输入数据的列表。
# 2. 如果有多个进程，则将输入数据序列化为一个Tensor对象。如果输入数据已经是一个Tensor对象，则将其展平为一维Tensor。
# 3. 接下来，获取每个进程的Tensor大小。首先创建一个本地大小的Tensor，然后使用`dist.all_gather`函数将各个进程的大小信息收集到一个列表中。
# 4. 确定最大的Tensor大小，用于创建接收数据的Tensor列表。如果本地大小不等于最大大小，则使用padding将本地Tensor填充到最大大小。
# 5. 使用`dist.all_gather`函数将各个进程的Tensor收集到Tensor列表中。
# 6. 最后，根据各个进程的大小信息，将收集到的Tensor转换为相应的数据对象。如果输入数据不是Tensor，则将Tensor转换为numpy数组，并使用pickle反序列化为数据对象。如果输入数据是Tensor，则根据原始大小信息将Tensor重新调整为原始形状。
# 7. 返回收集到的数据列表。

# 这个函数的作用是在分布式训练中收集来自各个进程的数据，以便进行后续的处理和分析。它可以用于收集模型参数、梯度等数据，以便进行全局同步或其他操作。

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to("cuda")
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).cuda().to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).cuda().to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list

# 这是一个名为`ImageLogger`的回调类，用于在训练过程中记录图像和相关信息。该类继承自`Callback`类，并重写了`on_train_batch_end`和`on_validation_batch_end`方法。

# `ImageLogger`类的构造函数接受多个参数，包括`batch_frequency`（记录图像的频率）、`max_images`（每个批次记录的最大图像数）、`clamp`（是否对图像进行限幅）、`increase_log_steps`（是否增加记录步骤的数量）、`rescale`（是否对图像进行重新缩放）、`disabled`（是否禁用图像记录）、`log_on_batch_idx`（是否在每个批次索引上记录图像）、`log_first_step`（是否在第一步记录图像）和`log_images_kwargs`（用于记录图像的其他参数）。

# `ImageLogger`类还定义了私有方法`_testtube`，用于在TestTubeLogger中记录图像。该方法接受`pl_module`（PyTorch-Lightning模块）、`images`（图像数据）、`batch_idx`（批次索引）和`split`（数据集划分）作为参数，在TestTubeLogger中添加图像。
# `ImageLogger`类还定义了私有方法`log_local`，用于在本地存储图像和相关信息。该方法接受`save_dir`（保存目录）、`split`（数据集划分）、`images`（图像数据）、`prompts`（提示信息）、`global_step`（全局步骤）、`current_epoch`（当前轮次）和`batch_idx`（批次索引）作为参数，将图像和相关信息保存到本地文件。
# `ImageLogger`类还定义了方法`log_img`，用于记录图像。该方法接受`pl_module`（PyTorch-Lightning模块）、`batch`（批次数据）、`batch_idx`（批次索引）和`split`（数据集划分，默认为"train"）作为参数。在满足一定条件时，该方法会调用`pl_module`的`log_images`方法获取图像数据，并调用`log_local`方法将图像和相关信息保存到本地文件。然后，根据不同的日志记录器类型，调用相应的方法将图像添加到日志中。
# `ImageLogger`类还定义了方法`check_frequency`，用于检查是否满足记录图像的频率。该方法接受`check_idx`（检查索引）作为参数，根据`batch_freq`和`log_steps`的设置判断是否满足记录图像的条件。
# 最后，`ImageLogger`类重写了`on_train_batch_end`和`on_validation_batch_end`方法，在训练和验证过程中调用`log_img`方法记录图像。如果模块具有`calibrate_grad_norm`属性，并且满足一定条件，还会调用`log_gradients`方法记录梯度信息。

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(6, int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, prompts,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        names = {"reals": "before", "inputs": "after", "reconstruction": "before-vq", "samples": "after-gen"}
        # print(root)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=8)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                names[k])
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # print(path)
            Image.fromarray(grid).save(path)

        filename = "gs-{:06}_e-{:06}_b-{:06}_prompt.json".format(
            global_step,
            current_epoch,
            batch_idx)
        path = os.path.join(root, filename)
        with open(path, "w") as f:
            for p in prompts:
                f.write(f"{json.dumps(p)}\n")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or (split == "val" and batch_idx == 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            prompts = batch["edit"]["c_crossattn"][:self.max_images]
            prompts = [p for ps in all_gather(prompts) for p in ps]

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                images[k] = torch.cat(all_gather(images[k][:N]))
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images, prompts,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            if len(self.log_steps) > 0:
                self.log_steps.pop(0)
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


# 这是一个名为`CUDACallback`的自定义回调类，它继承自PyTorch Lightning的`Callback`类。这个类主要用于在训练过程中跟踪GPU的内存使用情况和训练时间，并在每个epoch结束时输出相关信息。
# 在`on_train_epoch_start`方法中，它重置了GPU内存使用的峰值计数器，并记录了当前时间作为开始时间。
# 在`on_train_epoch_end`方法中，它首先同步GPU，然后计算出GPU内存使用的峰值（以兆字节为单位）和epoch的训练时间。然后，它尝试使用`trainer.training_type_plugin.reduce`方法来在分布式训练中对这些值进行归约操作。最后，它使用`rank_zero_info`函数将平均epoch时间和平均峰值内存打印出来。
# 需要注意的是，这个类依赖于一些其他的函数和对象，比如`torch.cuda.reset_peak_memory_stats`、`torch.cuda.synchronize`、`torch.cuda.max_memory_allocated`和`rank_zero_info`。你需要确保这些函数和对象在代码中已经定义或导入。

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

# 这段代码是一个训练脚本的主要部分。它首先解析命令行参数，并根据参数设置日志目录和检查点路径。然后，它加载配置文件并合并配置。
# 接下来，它根据配置实例化模型、训练器和回调函数。然后，它准备数据并进行训练或测试。最后，它打印出训练过程中的性能分析摘要。
# 这段代码使用了一些自定义的类和函数，例如get_parser()函数用于创建参数解析器，instantiate_from_config()函数用于根据配置实例化对象，
# SetupCallback类用于设置日志目录和检查点路径，ImageLogger类用于记录图像日志，LearningRateMonitor类用于记录学习率日志，CUDACallback类用于跟踪GPU的内存使用情况和训练时间。
# 整个训练过程中，还使用了一些PyTorch Lightning的功能，例如Trainer类用于管理训练过程，DataModule类用于准备和加载数据，ModelCheckpoint类用于保存模型检查点，WandbLogger类和TestTubeLogger类用于记录日志。


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    assert opt.name
    cfg_fname = os.path.split(opt.base[0])[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    nowname = f"{cfg_name}_{opt.name}"
    logdir = os.path.join(opt.logdir, nowname)
    ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    resume = False

    if os.path.isfile(ckpt):
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
        resume = True

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        if resume:
            # By default, when finetuning from Stable Diffusion, we load the EMA-only checkpoint to initialize all weights.
            # If resuming InstructPix2Pix from a finetuning checkpoint, instead load both EMA and non-EMA weights.
            config.model.params.load_ema = True

        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        print(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {
                "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    'save_top_k': -1,
                    'every_n_train_steps': 1000,
                    'save_weights_only': True
                }
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, plugins=DDPPlugin(find_unused_parameters=False), **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
