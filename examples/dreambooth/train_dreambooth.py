import argparse
import heapq
import hashlib
import itertools
import random
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import hflip
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default=None,
        required=False,
        help="The path of a file with filename-prompt pairs",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--save_width",
        type=int,
        default=512,
        help="The width for save sample.",
    )
    parser.add_argument(
        "--save_height",
        type=int,
        default=512,
        help="The height for save sample.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup", "onecycle"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_last_step", type=int, default=-1, help="Number of steps already computed."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--tag_dropout", type=float, default=0.95, help="Chance for remaining tags to be kept, for each tag")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--use_penultimate", action="store_true", help="Use penultimate hidden layer of text encoder output instead of the final layer")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        center_crop=False,
        prompt_file_path=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = []

        for concept in concepts_list:
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
            self.instance_images_path.extend(inst_img_path)

        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if prompt_file_path is not None:
            if not Path(prompt_file_path).exists():
                raise ValueError("Prompt file doesn't exists.")
            self.filename_to_prompt = {}
            with open(prompt_file_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    name, *rest = line.strip().split(":")
                    tag_str = ":".join(rest)
                    self.filename_to_prompt[name.strip()] = tag_str
        else:
            self.filename_to_prompt = None

        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                # transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _get_random_crop_box(self, im, multiple_of=8):
        new_w = im.width // multiple_of * multiple_of
        new_h = im.height // multiple_of * multiple_of
        dw = random.choice(range(0, im.width - new_w + 1))
        dh = random.choice(range(0, im.height - new_h + 1))
        return dw, dh, new_w - dw, new_h - dh

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index % self.num_instance_images]

        example["image_path"] = os.path.basename(image_path[0])
        prompt = image_path[1]
        if self.filename_to_prompt:
            name = os.path.basename(image_path[0])
            # Use same caption for flipped images
            if name in self.filename_to_prompt:
                prompt = self.filename_to_prompt[name]

        instance_image = Image.open(image_path[0]).convert("RGB")
        instance_image = instance_image.crop(self._get_random_crop_box(instance_image))
        if (torch.rand(1) > .5):
            instance_image = hflip(instance_image)
        example["instance_images"] = self.image_transforms(instance_image)
        example["prompt"] = prompt

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, vae, text_encoder, tokenizer, accelerator, weight_dtype, 
            source_dataloader, args):
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.source_dataloader = source_dataloader
        self.args = args

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        image_latent = self.latents_cache[index]
        self.latents_cache[index] = None
        text_latent = self.text_encoder_cache[index]
        self.text_encoder_cache[index] = None
        return image_latent, text_latent, self.paths[index] if self.paths else None

    def clear_latents(self):
        self.latents_cache = []
        self.text_encoder_cache = []
        self.paths = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cache_latents(self, limit=-1):
        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        if not self.args.train_text_encoder:
            self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        self.latents_cache = []
        self.text_encoder_cache = []
        self.paths = []
        total = limit if limit > -1 and limit < len(self.source_dataloader) else None
        for batch in tqdm(self.source_dataloader, desc="Caching latents", total=total):
            if limit > -1 and len(self) >= limit:
                break
            try_count = 0
            done = False
            while not done:
                try:
                    with torch.no_grad():
                        batch["pixel_values"] = batch["pixel_values"].to(self.accelerator.device, non_blocking=True, dtype=self.weight_dtype)
                        
                        self.latents_cache.append(self.vae.encode(batch["pixel_values"]).latent_dist)
                        self.paths += batch["paths"]
                        if self.args.train_text_encoder:
                            self.text_encoder_cache.append(batch["prompts"])
                        else:
                            self.text_encoder_cache.append(self._prompts_to_tokens(batch["prompts"]))
                        done = True
                except RuntimeError as e:
                    logger.exception("Inside latent caching")
                    # try up to 3 times
                    if try_count < 3 and "out of memory" in str(e):
                        if len(self.latents_cache) > len(self.text_encoder_cache):
                            self.latents_cache.pop()
                        try_count += 1
                        continue
                    raise e
        self.vae.cpu()
        if not self.args.train_text_encoder:
            self.text_encoder.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning("len: %s", len(self))

    def _prompts_to_tokens(self, prompts):
        prompts = [self._dropout_tags(p, per_tag_chance=self.args.tag_dropout) for p in prompts]
        input_ids = self.tokenizer(
                prompts,
                padding=True,
                truncation=False,
                return_tensors="pt"
            ).input_ids.to(self.accelerator.device, non_blocking=True)
        out = encode_tokens(self.text_encoder, input_ids.long(), self.args.use_penultimate)
        return out

    def _dropout_tags(self, prompt, per_tag_chance):
        tags = [t.strip() for t in prompt.split(",")]
        random.shuffle(tags)
        exceptions = []
        rest = []
        is_exception = lambda tag: tag.startswith("by ")
        for tag in tags:
            if is_exception(tag):
                exceptions.append(tag)
            else:
                rest.append(tag)
        for i in range(len(rest)):
            if torch.rand(1) >= per_tag_chance:
                break
        final = exceptions + rest[:i]
        random.shuffle(final)
        return ",".join(final)


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def encode_tokens_limited(text_encoder: CLIPTextModel, input_ids: torch.Tensor, use_penultimate: bool = True):
    if use_penultimate:
        hidden_state = text_encoder(input_ids, return_dict=True, output_hidden_states=True).hidden_states[-2]
        return text_encoder.text_model.final_layer_norm(hidden_state)
    else:
        return text_encoder(input_ids)[0]

# Bypass CLIP input limit by running multiple batches and concatenating
# Format of input_ids is beginning-of-string token, up to 75 tokens, end-of-string token, for 77 max
# todo fix batches?
def encode_tokens(text_encoder: CLIPTextModel, input_ids: torch.Tensor, use_penultimate: bool = True):
    first = input_ids[0]
    if len(first) > 77:
        bos, ids, eos = torch.split(first, [1, len(first)-2, 1])
        batches = torch.split(ids, 75)
        first = torch.unsqueeze(torch.cat((bos, batches[0], eos)), 0)
        out = encode_tokens_limited(text_encoder, first, use_penultimate=use_penultimate)
        for batch in batches[1:]:
            next_out = encode_tokens_limited(text_encoder, torch.unsqueeze(torch.cat((bos, batch, eos)), 0), use_penultimate=use_penultimate)
            out = torch.cat((out, next_out), axis=-2)
    else:
        out = encode_tokens_limited(text_encoder, input_ids, use_penultimate=use_penultimate)
    return out

def load_models(args):
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    
    return tokenizer, text_encoder, vae, unet

def reload_models(args, accelerator, optimizer, train_dataloader, lr_scheduler):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    text_encoder = None
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.train_text_encoder:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        if args.gradient_checkpointing and args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    return unet, text_encoder, optimizer, train_dataloader, lr_scheduler

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def debug_gpu(namespace):
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logger.warning("type:%s, size:%s ", type(obj), obj.size())
                referrers = gc.get_referrers(obj)
                for r in referrers:
                    if torch.is_tensor(r) or (hasattr(r, 'data') and torch.is_tensor(r.data)):
                        logger.warning("referrer size: %s, referrer names: %s", r.size(), namestr(r, namespace))
                    else:
                        logger.warning("referrer type: %s, referrer names: %s", type(r), namestr(r, namespace))
        except:
            pass

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "0", args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "instance_data_dir": args.instance_data_dir,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    tokenizer, text_encoder, vae, unet = load_models(args)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        prompt_file_path=args.prompt_file_path,
    )
    
    def prompts_to_tokens(prompts):
        prompts = [dropout_tags(p, per_tag_chance=args.tag_dropout) for p in prompts]
        input_ids = tokenizer(
                prompts,
                padding=True,
                truncation=False,
                return_tensors="pt"
            ).input_ids.to(accelerator.device, non_blocking=True)
        out = encode_tokens(text_encoder, input_ids.long(), args.use_penultimate)
        return out

    def dropout_tags(prompt, per_tag_chance=args.tag_dropout):
        tags = [t.strip() for t in prompt.split(",")]
        random.shuffle(tags)
        exceptions = []
        rest = []
        is_exception = lambda tag: tag.startswith("by ")
        for tag in tags:
            if is_exception(tag):
                exceptions.append(tag)
            else:
                rest.append(tag)
        for i in range(len(rest)):
            if torch.rand(1) >= per_tag_chance:
                break
        final = exceptions + rest[:i]
        random.shuffle(final)
        return ",".join(final)

    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]
        paths = [example["image_path"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        prompts = [example["prompt"] for example in examples]

        batch = {
            "prompts": prompts,
            "pixel_values": pixel_values,
            "paths": paths,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    if not args.not_cache_latents:
        train_dataset = LatentsDataset(vae, text_encoder, tokenizer, accelerator, weight_dtype, source_dataloader=train_dataloader, args=args)
        limit = -1 if args.max_train_steps is None else args.max_train_steps
        train_dataset.cache_latents(limit=limit)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
        overrode_max_train_steps = True


    if args.lr_scheduler == "onecycle":
        total_steps = args.max_train_steps
        if args.lr_last_step > 0:
            total_steps += args.lr_last_step
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.learning_rate,
            total_steps=total_steps,
        )
        if args.lr_last_step > 0:
            for _ in range(args.lr_last_step):
                lr_scheduler.step()
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path, subfolder=None if args.pretrained_vae_name_or_path else "vae"),
                safety_checker=None,
                scheduler=scheduler,
                torch_dtype=torch.float16,
            )
            save_dir = os.path.join(args.output_dir, f"{step}")
            pipeline.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

            if args.save_sample_prompt is not None:
                pipeline = pipeline.to(accelerator.device)
                g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                pipeline.set_progress_bar_config(disable=True)
                sample_dir = os.path.join(save_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                with torch.autocast("cuda"), torch.inference_mode():
                    for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            width=args.save_width,
                            height=args.save_height,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(os.path.join(sample_dir, f"{i}.png"))
                    g_cuda.seed()
                    for i in tqdm(range(args.n_save_sample), desc="Generating random samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            width=args.save_width,
                            height=args.save_height,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(os.path.join(sample_dir, f"rand-{i}.png"))
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[*] Weights saved at {save_dir}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    loss_heap = []
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    first_run = True
    try:
        for epoch in range(args.num_train_epochs):
            if not args.not_cache_latents and not first_run:
                try_count = 0
                while (True):
                    try:
                        steps_left = args.max_train_steps - global_step
                        train_dataset.cache_latents(limit=steps_left)
                        break
                    except RuntimeError as e:
                        logger.exception("During latent caching")
                        # try up to 3 times
                        if try_count < 3 and "out of memory" in str(e):
                            train_dataset.clear_latents()
                            try_count += 1
                            continue
                        raise e

            first_run = False
            unet.train()
            for step, batch in enumerate(train_dataloader):
                try:
                    with accelerator.accumulate(unet):
                        # Convert images to latent space
                        with torch.no_grad():
                            if not args.not_cache_latents:
                                latent_dist = batch[0][0]
                            else:
                                latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                            latents = latent_dist.sample() * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        with text_enc_context:
                            if not args.not_cache_latents:
                                if args.train_text_encoder:
                                    encoder_hidden_states = prompts_to_tokens(batch[0][1])
                                else:
                                    encoder_hidden_states = batch[0][1]
                            else:
                                encoder_hidden_states = prompts_to_tokens(batch["prompts"])

                        # Predict the noise residual
                        encoder_hidden_states.to(accelerator.device, dtype=weight_dtype)
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        encoder_hidden_states.cpu()

                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(unet.parameters(), text_encoder.parameters())
                                if args.train_text_encoder
                                else unet.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if not math.isnan(loss.item()):
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            loss_avg.update(loss.detach_(), bsz)
                        else:
                            logger.warning("nan loss. noise=%s, noise_pred=%s, paths=%s",noise, noise_pred, [str(b[2]) for b in batch])
                            raise Exception("Nan loss, cannot continue")
                        lr_scheduler.step()

                    if not global_step % args.log_interval:
                        logs = {"loss": loss.item(), "loss_avg": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)

                    loss_heap.append({"loss": loss.item(), "paths": [str(b[2]) for b in batch]})
                    
                except RuntimeError as e:
                    logger.exception("During accumulation")
                    if "out of memory" in str(e):
                        logger.warning("paths %s",[str(b[2]) for b in batch])
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        raise e
                
                if global_step > args.save_min_steps and not global_step % args.save_interval:
                    save_weights(global_step)

                progress_bar.update(1)
                global_step += 1

                if global_step >= args.max_train_steps:
                    break

            accelerator.wait_for_everyone()
            if global_step >= args.max_train_steps:
                break
    
    except RuntimeError as e:
        logger.exception("During outer loop")
        if "out of memory" in str(e):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    except KeyboardInterrupt as e:
        logger.warning("KeyboardInterrupt")

    save_weights(global_step)

    accelerator.end_training()
    try:
        largest = heapq.nlargest(20, loss_heap, key=lambda x : x["loss"])
        for l in largest:
            logger.warning(str(l))
    except:
        pass



if __name__ == "__main__":
    main()
