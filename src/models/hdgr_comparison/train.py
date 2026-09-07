"""
Training Code for CLIP-SF
Modified Version:
1. Add Clash proxy support
2. Skip unused T5 tokenizer in HDGR block-denoising mode
3. Support offline/local-files-only mode
4. Fix non-distributed barrier/init issues
"""

# Standard library
import argparse
import logging
import os
import random
import gc
import json
import math
from pathlib import Path

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb


# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
)
from data.mbeir_dataset import MBEIRDictInstructioneDataset
from models.utils import cosine_warmup_scheduler

from models.uniir_clip import utils
from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion
from models.hdgr_comparison.model_factory import (
    get_generative_retriever_class,
    uses_t5_tokenizer,
    is_hdgr_block_denoising_config,
    is_gpt_hdgr_config,
    is_genius_ar_config,
)
from models.hdgr_comparison.engine import train_one_epoch, eval_engine


# -------------------------
# Logger
# -------------------------
logger = logging.getLogger()


# -------------------------
# Env / Proxy / HF helpers
# -------------------------
def setup_clash_proxy(enable=False, proxy_url="http://127.0.0.1:7890", hf_home=None):
    """
    为 requests / urllib / huggingface_hub / transformers 设置 Clash 代理。
    """
    if not enable:
        if hf_home:
            os.environ["HF_HOME"] = hf_home
        return

    proxy_keys = [
        "HTTP_PROXY", "HTTPS_PROXY",
        "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy",
    ]
    for k in proxy_keys:
        os.environ[k] = proxy_url

    # 本地地址不走代理
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

    # HF 超时适当放大
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    print(f"[Proxy] Clash proxy enabled: {proxy_url}")
    if hf_home:
        print(f"[HF] HF_HOME set to: {hf_home}")


def setup_offline_mode(local_files_only=False, hf_home=None):
    """
    开启本地离线模式。
    """
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print("[HF] Offline mode enabled (local_files_only=True)")


def safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_t5_tokenizer(local_t5_dir=None, model_name="google-t5/t5-small", model_max_length=42, local_files_only=False):
    """Legacy T5 tokenizer loader.  Only used by the old T5 generator path."""
    from transformers import T5TokenizerFast

    if local_t5_dir and os.path.isdir(local_t5_dir):
        print(f"[Tokenizer] Loading legacy T5 tokenizer from local dir: {local_t5_dir}")
        return T5TokenizerFast.from_pretrained(
            local_t5_dir,
            model_max_length=model_max_length,
            local_files_only=True,
        )

    print(f"[Tokenizer] Loading legacy T5 tokenizer from remote/local cache: {model_name}")
    return T5TokenizerFast.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        local_files_only=local_files_only,
    )


# -------------------------
# Utils
# -------------------------
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了更稳定的复现，benchmark 关掉
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]


def create_optimizer(gain_or_bias_params, rest_params, config):
    """Create optimizer from trainer_config.

    Backward-compatible defaults keep the original AdamW setup.  New configs can
    request official MaskGIT-style Adam settings with:

        trainer_config:
          optimizer: adam
          learning_rate: 2e-4
          adam_beta1: 0.9
          adam_beta2: 0.96
          weight_decay: 0.0
    """
    trainer_cfg = config.trainer_config
    optimizer_name = str(getattr(trainer_cfg, "optimizer", "adamw")).lower()
    adam_beta1 = float(getattr(trainer_cfg, "adam_beta1", 0.9))
    adam_beta2 = float(getattr(trainer_cfg, "adam_beta2", 0.98))
    adam_eps = float(getattr(trainer_cfg, "adam_eps", 1.0e-6))
    weight_decay = float(getattr(trainer_cfg, "weight_decay", 1.0e-4))
    gain_or_bias_weight_decay = float(getattr(trainer_cfg, "gain_or_bias_weight_decay", 0.0))

    if optimizer_name == "adam":
        optimizer_cls = optim.Adam
    elif optimizer_name == "adamw":
        optimizer_cls = optim.AdamW
    else:
        raise ValueError(f"Unsupported trainer_config.optimizer={optimizer_name!r}; use 'adam' or 'adamw'.")

    return optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": gain_or_bias_weight_decay},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=trainer_cfg.learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )


def _as_float(value, default=None):
    try:
        if value is None:
            return default
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default




def _stats_are_finite(stats):
    if not stats:
        return True
    for _k, _v in stats.items():
        try:
            _f = float(_v)
        except Exception:
            continue
        if math.isnan(_f) or math.isinf(_f):
            print(f"[soundstorm-nan-guard] non-finite train stat {_k}={_v}; will not save checkpoint")
            return False
    return True

def _checkpoint_base_name(config):
    return str(config.model.short_name).lower()


def _checkpoint_dir(config):
    ckpt_config = config.model.ckpt_config
    checkpoint_dir = os.path.join(config.genir_dir, ckpt_config.ckpt_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def _checkpoint_path(config, kind="latest"):
    model_name = _checkpoint_base_name(config)
    return os.path.join(_checkpoint_dir(config), f"{model_name}_{kind}.pth")


def _write_best_metadata(config, best_metric_name, best_metric_value, best_epoch, latest_metric_value=None, latest_epoch=None):
    metadata_path = os.path.join(_checkpoint_dir(config), f"{_checkpoint_base_name(config)}_best_R_at_1.json")
    metadata = {
        "selection_metric": best_metric_name,
        "best_R_at_1": best_metric_value,
        "best_epoch": best_epoch,
        "latest_R_at_1": latest_metric_value,
        "latest_epoch": latest_epoch,
        "best_checkpoint": os.path.basename(_checkpoint_path(config, "best_R_at_1")),
        "latest_checkpoint": os.path.basename(_checkpoint_path(config, "latest")),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)




class SoundStormReduceLROnPlateauWithWarmup:
    """SoundStorm/VQ-Diffusion scheduler adapted from SoundStorm S2.

    The original SoundStorm S2 configs use:
      target: soundstorm.s2.engine.lr_scheduler.ReduceLROnPlateauWithWarmup
      factor: 0.5, patience: 25000, threshold: 1e-1, threshold_mode: rel,
      min_lr: 2e-6, warmup_lr: 0.9e-3, warmup: 800.

    This lightweight implementation keeps the same per-iteration semantics while
    matching the scheduler API used by this training loop.
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.5,
        patience=25000,
        threshold=1.0e-1,
        threshold_mode="rel",
        cooldown=0,
        min_lr=2.0e-6,
        eps=1.0e-8,
        warmup_lr=0.9e-3,
        warmup=800,
        verbose=False,
    ):
        if factor >= 1.0:
            raise ValueError("factor should be < 1.0")
        if mode not in {"min", "max"}:
            raise ValueError(f"unknown mode={mode}")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"unknown threshold_mode={threshold_mode}")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.cooldown = int(cooldown)
        self.cooldown_counter = 0
        self.eps = float(eps)
        self.warmup = int(warmup)
        self.verbose = bool(verbose)
        self.last_epoch = 0
        self.num_bad_epochs = 0
        self.best = float("inf") if mode == "min" else -float("inf")
        if isinstance(min_lr, (list, tuple)):
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [float(min_lr)] * len(optimizer.param_groups)
        if isinstance(warmup_lr, (list, tuple)):
            self.warmup_lrs = list(warmup_lr)
        else:
            self.warmup_lrs = [float(warmup_lr)] * len(optimizer.param_groups)
        self._prepare_warmup_steps()

    def _prepare_warmup_steps(self):
        if self.warmup > self.last_epoch:
            curr_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            remain = max(1, self.warmup - self.last_epoch)
            self.warmup_lr_steps = [
                max(0.0, (self.warmup_lrs[i] - curr_lrs[i]) / float(remain))
                for i in range(len(curr_lrs))
            ]
        else:
            self.warmup_lr_steps = None

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, current, best):
        if self.mode == "min" and self.threshold_mode == "rel":
            return current < best * (1.0 - self.threshold)
        if self.mode == "min" and self.threshold_mode == "abs":
            return current < best - self.threshold
        if self.mode == "max" and self.threshold_mode == "rel":
            return current > best * (1.0 + self.threshold)
        return current > best + self.threshold

    def _increase_lr(self, epoch):
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = max(old_lr + self.warmup_lr_steps[i], self.min_lrs[i])
            group["lr"] = new_lr
            if self.verbose:
                print(f"SoundStorm scheduler iter {epoch}: increasing lr of group {i} to {new_lr:.4e}")

    def _reduce_lr(self, epoch):
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr
                if self.verbose:
                    print(f"SoundStorm scheduler iter {epoch}: reducing lr of group {i} to {new_lr:.4e}")

    def step(self, metrics=None):
        # SoundStorm calls scheduler.step(loss) every iteration.  If a metric is
        # unavailable, use zero only to keep warmup iteration counting valid.
        current = 0.0 if metrics is None else float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch <= self.warmup and self.warmup_lr_steps is not None:
            self._increase_lr(epoch)
            return
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def state_dict(self):
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "cooldown_counter": self.cooldown_counter,
            "eps": self.eps,
            "warmup": self.warmup,
            "warmup_lrs": self.warmup_lrs,
            "warmup_lr_steps": self.warmup_lr_steps,
            "min_lrs": self.min_lrs,
            "last_epoch": self.last_epoch,
            "num_bad_epochs": self.num_bad_epochs,
            "best": self.best,
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._prepare_warmup_steps()


def create_scheduler(optimizer, config, total_steps):
    trainer_cfg = config.trainer_config
    scheduler_name = str(getattr(trainer_cfg, "scheduler", "cosine_warmup")).lower()
    if scheduler_name in {"soundstorm", "soundstorm_reduce_on_plateau_warmup", "reduce_on_plateau_warmup"}:
        scheduler = SoundStormReduceLROnPlateauWithWarmup(
            optimizer,
            mode=str(getattr(trainer_cfg, "soundstorm_scheduler_mode", "min")),
            factor=float(getattr(trainer_cfg, "soundstorm_scheduler_factor", 0.5)),
            patience=int(getattr(trainer_cfg, "soundstorm_scheduler_patience", 25000)),
            threshold=float(getattr(trainer_cfg, "soundstorm_scheduler_threshold", 1.0e-1)),
            threshold_mode=str(getattr(trainer_cfg, "soundstorm_scheduler_threshold_mode", "rel")),
            cooldown=int(getattr(trainer_cfg, "soundstorm_scheduler_cooldown", 0)),
            min_lr=float(getattr(trainer_cfg, "soundstorm_min_lr", 2.0e-6)),
            eps=float(getattr(trainer_cfg, "soundstorm_scheduler_eps", 1.0e-8)),
            warmup_lr=float(getattr(trainer_cfg, "soundstorm_warmup_lr", 0.9e-3)),
            warmup=int(getattr(trainer_cfg, "warmup_steps", 800)),
            verbose=bool(getattr(trainer_cfg, "soundstorm_scheduler_verbose", False)),
        )
        print(
            "[generator train] scheduler: soundstorm_reduce_on_plateau_warmup "
            f"base_lr={optimizer.param_groups[0]['lr']:.6g}, "
            f"warmup_lr={float(getattr(trainer_cfg, 'soundstorm_warmup_lr', 0.9e-3)):.6g}, "
            f"warmup_steps={int(getattr(trainer_cfg, 'warmup_steps', 800))}, "
            f"factor={float(getattr(trainer_cfg, 'soundstorm_scheduler_factor', 0.5))}, "
            f"patience={int(getattr(trainer_cfg, 'soundstorm_scheduler_patience', 25000))}, "
            f"min_lr={float(getattr(trainer_cfg, 'soundstorm_min_lr', 2.0e-6))}"
        )
        return scheduler
    warmup_steps = int(getattr(config.trainer_config, "warmup_steps", 0))
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=warmup_steps,
        total_epochs=total_steps,
    )
    print(f"[generator train] scheduler: cosine_warmup warmup_steps={warmup_steps}, total_steps={total_steps}")
    return scheduler


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    scaler,
    config,
    kind="latest",
    metric_name="train_R_at_1",
    metric_value=None,
    best_metric_value=None,
    best_epoch=None,
):
    checkpoint_path = _checkpoint_path(config, kind)

    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
        "selection_metric": metric_name,
        "R_at_1": metric_value,
        "best_R_at_1": best_metric_value,
        "best_epoch": best_epoch,
    }

    torch.save(save_obj, checkpoint_path)
    if kind == "latest":
        print(f"Saved latest checkpoint to {checkpoint_path} (Overwrote previous latest checkpoint)")
    else:
        print(
            f"Saved best R_at_1 checkpoint to {checkpoint_path} "
            f"(epoch={epoch}, {metric_name}={metric_value:.6f})"
        )
    return checkpoint_path


def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats


# -------------------------
# Train loop
# -------------------------
def train(
    train_loader,
    val_loader,
    model,
    model_without_ddp,
    clip_model,
    optimizer,
    scheduler,
    scaler,
    config,
    start_epoch,
    best_r_at_1=-float("inf"),
    best_epoch=None,
):
    gpu_id = config.dist_config.gpu_id
    is_distributed_mode = bool(config.dist_config.distributed_mode)
    global_step = 0
    model.zero_grad()

    if start_epoch != 0:
        print(f"Resuming training from epoch {start_epoch}")
    if best_epoch is not None and best_r_at_1 != -float("inf"):
        print(f"[best-checkpoint] Resume tracker: best_R_at_1={best_r_at_1:.6f} at epoch={best_epoch}")
    else:
        print("[best-checkpoint] Tracking best checkpoint by train R_at_1.")

    for epoch in range(start_epoch, config.trainer_config.num_train_epochs):
        if is_distributed_mode and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            clip_model,
            train_loader,
            optimizer,
            epoch,
            gpu_id,
            scheduler,
            global_step,
            scaler,
            config,
        )
        if not _stats_are_finite(train_stats):
            raise RuntimeError("Non-finite training stats detected; aborting before checkpoint save. Restore from best_R_at_1 or previous clean epoch.")
        gc.collect()

        eval_freq = config.evaluator.eval_freq
        save_freq = max(1, int(getattr(config.evaluator, "save_freq", eval_freq)))
        save_best = bool(getattr(config.evaluator, "save_best", True))

        val_status = None
        if val_loader is not None and epoch % eval_freq == 0 and epoch >= config.evaluator.eval_start:
            val_status = eval_engine(model, clip_model, val_loader, gpu_id, config)

        # All generator variants report R_at_1 during training.  Save both the
        # latest epoch and the best epoch according to train R_at_1 so model
        # selection is consistent across HDGR / GPT / GENIUS_AR / BD3 variants.
        current_r_at_1 = _as_float(train_stats.get("R_at_1") if train_stats else None)
        metric_name = "train_R_at_1"
        final_epoch = epoch + 1 == config.trainer_config.num_train_epochs
        should_save = (epoch + 1) % save_freq == 0 or final_epoch
        if is_main_process() and should_save:
            improved = current_r_at_1 is not None and current_r_at_1 > best_r_at_1
            if improved and save_best:
                best_r_at_1 = current_r_at_1
                best_epoch = epoch
            save_checkpoint(
                model_without_ddp, optimizer, scheduler, epoch, scaler, config,
                kind="latest", metric_name=metric_name, metric_value=current_r_at_1,
                best_metric_value=best_r_at_1 if best_r_at_1 != -float("inf") else None,
                best_epoch=best_epoch,
            )
            if improved and save_best:
                save_checkpoint(
                    model_without_ddp, optimizer, scheduler, epoch, scaler, config,
                    kind="best_R_at_1", metric_name=metric_name, metric_value=current_r_at_1,
                    best_metric_value=best_r_at_1, best_epoch=best_epoch,
                )
            _write_best_metadata(
                config, metric_name,
                None if best_r_at_1 == -float("inf") else best_r_at_1,
                best_epoch, latest_metric_value=current_r_at_1, latest_epoch=epoch,
            )

        if val_status is not None:
            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch=best_epoch)
        else:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch=best_epoch)
        if best_r_at_1 != -float("inf"):
            log_stats["best_R_at_1"] = best_r_at_1

        if is_main_process() and bool(config.wandb_config.enabled):
            wandb.log(log_stats)

        safe_barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------
# Main
# -------------------------
def main(config, args):
    is_distributed_mode = bool(config.dist_config.distributed_mode)

    # Set up seed
    seed = config.seed + utils.get_rank()
    set_seed(seed)

    cudnn.benchmark = False

    # Initialize and load model
    print("Creating CLIP-SF model...")
    model_config = config.model

    pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading/loading CLIP model to/from {pretrained_clip_model_dir}...")

    clip_model = CLIPNoFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    clip_model.float()
    clip_model.eval()

    # HDGR and GPT baselines do not need an external T5 tokenizer/backbone.
    # Keep the legacy loader only for old T5 generator experiments.
    if not uses_t5_tokenizer(config):
        seq2seq_tokenizer = None
        if is_gpt_hdgr_config(config):
            print("[Tokenizer] GPT-HDGR backbone baseline: using internal code tokenizer; skipping T5 tokenizer.")
        elif is_genius_ar_config(config):
            print("[Tokenizer] GENIUS_AR official T5-style baseline: using internal code tokenizer; skipping external T5 tokenizer.")
        else:
            print("[Tokenizer] HDGR block-denoising mode: skipping unused T5 tokenizer.")
    else:
        seq2seq_tokenizer = build_t5_tokenizer(
            local_t5_dir=args.t5_local_dir,
            model_name=args.t5_model_name,
            model_max_length=42,
            local_files_only=True,
        )

    RetrieverClass = get_generative_retriever_class(config)
    model = RetrieverClass(config=config, tokenizer=seq2seq_tokenizer)

    # Move after distributed rank/device is known below.  Avoid hard-coded cuda:0
    # so this entry works for CPU smoke tests, single GPU, and torchrun.

    # Optimizer / scaler
    exclude_condition = lambda n, p: (
        p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    )
    include_condition = lambda n, p: not exclude_condition(n, p)

    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)
    scaler = GradScaler()

    # Load CLIP pretrained
    pretrained_config = model_config.pretrained_config
    if pretrained_config.using_pretrained:
        pretrained_path = os.path.join(
            config.genir_dir,
            pretrained_config.pretrained_dir,
            pretrained_config.pretrained_name,
        )
        assert os.path.exists(pretrained_path), f"Checkpoint file {pretrained_path} does not exist."
        logger.info(f"Loading CLIP checkpoint from {pretrained_path}")
        checkpoint = torch.load(
            pretrained_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        clip_model.load_state_dict(checkpoint["model"])

    # Optional weight-only initialization for two-stage generator training.
    # This is intentionally separate from resume_training: it loads Stage-3A
    # weights into Stage-3B, but starts a fresh optimizer/scheduler and epoch 0.
    ckpt_config = model_config.ckpt_config
    checkpoint = None
    if ckpt_config and (not getattr(ckpt_config, "resume_training", False)) and bool(getattr(ckpt_config, "init_from_checkpoint", False)):
        init_ckpt_path = getattr(ckpt_config, "init_ckpt_path", "")
        if init_ckpt_path:
            init_checkpoint_path = init_ckpt_path
            if not os.path.isabs(init_checkpoint_path):
                init_checkpoint_path = os.path.join(config.genir_dir, init_checkpoint_path)
        else:
            init_ckpt_dir = getattr(ckpt_config, "init_ckpt_dir", "")
            init_ckpt_name = getattr(ckpt_config, "init_ckpt_name", "")
            init_checkpoint_path = os.path.join(config.genir_dir, init_ckpt_dir, init_ckpt_name)
        assert os.path.exists(init_checkpoint_path), f"Init checkpoint file {init_checkpoint_path} does not exist."
        init_strict = bool(getattr(ckpt_config, "init_strict", False))
        print(f"Initializing GenerativeRetriever weights from {init_checkpoint_path} (strict={init_strict})")
        init_checkpoint = torch.load(init_checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        init_state = init_checkpoint.get("model", init_checkpoint)
        missing, unexpected = model.load_state_dict(init_state, strict=init_strict)
        print(f"Initialized from checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"  Missing keys preview: {missing[:10]}")
        if len(unexpected) > 0:
            print(f"  Unexpected keys preview: {unexpected[:10]}")

    # Resume retriever checkpoint, including epoch/optimizer state.
    if ckpt_config and getattr(ckpt_config, "resume_training", False):
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"Loading GenerativeRetriever checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)

    # Move model to GPU
    model.train()
    model = model.to(config.dist_config.gpu_id)
    model_without_ddp = model

    if is_distributed_mode:
        model = DDP(
            model,
            device_ids=[config.dist_config.gpu_id],
            find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # Dataset / Dataloader
    logger.info("Preparing dataset ...")
    logger.info(f"Loading dataset from {os.path.join(config.mbeir_data_dir, config.data_config.train_query_data_path)}...")

    img_preprocess_fn = clip_model.get_img_preprocess_fn()
    clip_tokenizer = clip_model.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    query_dict_dir = os.path.join(config.genir_dir, config.codebook_config.query_path)
    pool_dict_dir = os.path.join(config.genir_dir, config.codebook_config.pool_path)

    data_config = config.data_config
    train_dataset = MBEIRDictInstructioneDataset(
        mbeir_data_dir=config.mbeir_data_dir,
        query_data_path=data_config.train_query_data_path,
        cand_pool_path=data_config.train_cand_pool_path,
        query_instruct_path=data_config.query_instruct_path,
        query_dict_dir=query_dict_dir,
        pool_dict_dir=pool_dict_dir,
        # The HDGR generator conditions on precomputed query embeddings.
        # Instruction text has already been folded into *_instruction_* feature
        # dictionaries, so no seq2seq/T5 tokenizer is needed here.
        tokenizer=seq2seq_tokenizer,
        return_instruct=seq2seq_tokenizer is not None,
    )

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader_config.train_batch_size,
        num_workers=config.dataloader_config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
    )

    enable_eval = bool(config.evaluator.enable_eval)
    valid_loader = None

    if enable_eval:
        in_batch_val_dataset, in_batch_val_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=clip_tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.IN_BATCH_VAL,
        )

        in_batch_val_sampler = DistributedSampler(
            dataset=in_batch_val_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )

        valid_loader = DataLoader(
            dataset=in_batch_val_dataset,
            batch_size=config.dataloader_config.valid_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=True,
            sampler=in_batch_val_sampler,
            shuffle=False,
            collate_fn=in_batch_val_collector,
            drop_last=True,
        )
    else:
        print("In-batch validation is disabled.")

    # Scheduler
    t_total = (
        math.ceil(len(train_loader) / config.trainer_config.gradient_accumulation_steps)
        * config.trainer_config.num_train_epochs
    )
    scheduler = create_scheduler(optimizer, config, t_total)

    start_epoch = 0
    best_r_at_1 = -float("inf")
    best_epoch = None
    if ckpt_config.resume_training and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_r_at_1 = _as_float(checkpoint.get("best_R_at_1"), -float("inf"))
        best_epoch = checkpoint.get("best_epoch", None)
        if best_epoch is None and _as_float(checkpoint.get("R_at_1"), None) == best_r_at_1:
            best_epoch = checkpoint.get("epoch", None)

    safe_barrier()

    train(
        train_loader,
        valid_loader,
        model,
        model_without_ddp,
        clip_model,
        optimizer,
        scheduler,
        scaler,
        config,
        start_epoch,
        best_r_at_1=best_r_at_1,
        best_epoch=best_epoch,
    )


# -------------------------
# Worker
# -------------------------

def apply_cli_dotlist_overrides(config, unknown_args, rank=0):
    """Apply Hydra/OmegaConf-style CLI overrides passed after known argparse args.

    This project mainly uses argparse, but our experiment commands often use
    overrides such as `runtime.quantizer_path=...` or
    `model.ckpt_config.ckpt_name=...`.  Supporting a small dotlist layer keeps
    the train/eval scripts compatible with both styles.
    """
    if not unknown_args:
        return config

    dotlist = []
    ignored = []
    for item in unknown_args:
        if not isinstance(item, str):
            continue
        if "=" in item and not item.startswith("-"):
            # Historical alias used in our scripts; map it to the actual config key.
            if item.startswith("runtime.quantizer_path="):
                value = item.split("=", 1)[1]
                if not hasattr(config, "codebook_config"):
                    config.codebook_config = {}
                config.codebook_config.quantizer_path = value
                if rank == 0:
                    print(f"[CLI override] runtime.quantizer_path -> codebook_config.quantizer_path = {value}")
            else:
                dotlist.append(item)
        else:
            ignored.append(item)

    if dotlist:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist))
        if rank == 0:
            print(f"[CLI override] Applied dotlist overrides: {dotlist}")

    # If a dotlist created runtime.quantizer_path, keep codebook_config in sync.
    try:
        runtime_q = config.runtime.quantizer_path
    except Exception:
        runtime_q = None
    if runtime_q:
        if not hasattr(config, "codebook_config"):
            config.codebook_config = {}
        config.codebook_config.quantizer_path = runtime_q
        if rank == 0:
            print(f"[CLI override] Synced runtime.quantizer_path -> codebook_config.quantizer_path = {runtime_q}")

    if ignored and rank == 0:
        print(f"[CLI override] Ignored unknown non-dotlist args: {ignored}")
    return config


def main_worker(args):
    """
    torchrun 启动模式：
    每个进程由 torch.distributed.run 拉起，
    读取 LOCAL_RANK / RANK / WORLD_SIZE。
    非 torchrun 单卡模式下也能正常运行。
    """
    # 先设置 HF / proxy 环境
    setup_offline_mode(local_files_only=args.local_files_only, hf_home=args.hf_home)
    setup_clash_proxy(
        enable=args.use_clash_proxy,
        proxy_url=args.clash_proxy_url,
        hf_home=args.hf_home,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"[Rank {rank}] [Local Rank {local_rank}] Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir
    if args.quantizer_path:
        if not hasattr(config, "codebook_config"):
            config.codebook_config = {}
        config.codebook_config.quantizer_path = args.quantizer_path
        print(f"[Rank {rank}] Override quantizer_path -> {args.quantizer_path}")

    config = apply_cli_dotlist_overrides(config, getattr(args, "config_overrides", []), rank=rank)

    args.gpu = local_rank
    args.distributed = world_size > 1
    if hasattr(config.dist_config, "dist_url"):
        args.dist_url = config.dist_config.dist_url

    # 分布式初始化：只有 world_size > 1 才真正 init
    if world_size > 1:
        try:
            utils.init_distributed_mode(args)
        except Exception as e:
            print(f"[Warn] utils.init_distributed_mode failed: {e}")
            if not dist.is_initialized():
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://")
    else:
        torch.cuda.set_device(local_rank)

    config.dist_config.gpu_id = local_rank
    config.dist_config.distributed_mode = (world_size > 1)

    is_main = is_main_process()

    if is_main:
        logger_out_dir = os.path.join(config.genir_dir, config.logger_config.logger_out_dir)
        logger_out_path = os.path.join(logger_out_dir, config.logger_config.logger_out_file_name)
        os.makedirs(logger_out_dir, exist_ok=True)

        handlers = [logging.FileHandler(logger_out_path), logging.StreamHandler()]
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            level=logging.DEBUG,
            datefmt="%d-%m-%Y %H:%M:%S",
            handlers=handlers,
            force=True,
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logger.info(config)

    if bool(config.wandb_config.enabled) and is_main:
        load_dotenv()
        wandb_key = config.wandb_config.wandb_key
        wandb_project = config.wandb_config.wandb_project
        wandb_entity = os.environ.get("WANDB_ENTITY")

        if not wandb_key:
            raise ValueError("WANDB_API_KEY not found. Ensure it's set in the .env file.")

        wandb.login(key=wandb_key)
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=config.wandb_config.experiment_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    main(config, args)

    if bool(config.wandb_config.enabled) and is_main:
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="configs/structnar/train.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--genir_dir",
        type=str,
        default=str(project_root),
        help="Path to GENIUS directory.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default=str(project_root / "mbeir_data"),
        help="Path to mbeir dataset directory",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    # Proxy / HF / legacy tokenizer args
    parser.add_argument(
        "--use_clash_proxy",
        action="store_true",
        help="Enable Clash proxy for HuggingFace / CLIP downloads.",
    )
    parser.add_argument(
        "--clash_proxy_url",
        type=str,
        default="http://127.0.0.1:7890",
        help="Clash proxy url, e.g. http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default="",
        help="Optional HuggingFace cache directory, e.g. ~/.cache/huggingface",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force transformers/huggingface to load only local files.",
    )
    parser.add_argument(
        "--t5_local_dir",
        type=str,
        default="",
        help="Legacy only: local path of google-t5/t5-small tokenizer files.",
    )
    parser.add_argument(
        "--t5_model_name",
        type=str,
        default="google-t5/t5-small",
        help="Legacy only: remote HF model name for T5 tokenizer.",
    )
    parser.add_argument(
        "--quantizer_path",
        type=str,
        default="",
        help=(
            "Optional override for codebook_config.quantizer_path. "
            "Accepts either an absolute path or a path relative to --genir_dir."
        ),
    )

    args, unknown_args = parser.parse_known_args()
    args.config_overrides = unknown_args

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but training requires CUDA.")

    print("启动训练 (torchrun mode / single process mode)...")
    main_worker(args)
