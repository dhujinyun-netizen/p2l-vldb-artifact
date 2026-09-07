"""
Training Code for CLIP-SF
"""

# Standard library
import argparse
import logging
import os
import random
import clip 

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from omegaconf import OmegaConf
from dotenv import load_dotenv
import wandb
import pickle
from tqdm import tqdm
import gc

# Local modules or packages
from data.mbeir_data_utils import (
    build_mbeir_dataset_from_config,
    DatasetType,
    build_distributed_sampler_list,
    build_dataloader_list,
)

from data.preprocessing.utils import (
    format_string,
    hash_did,
    hash_qid,
    unhash_did,
    unhash_qid,
    get_mbeir_task_id,
)

from models.uniir_clip.engine import train_one_epoch, eval_engine
from models.uniir_clip import utils
from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion


# Set up logger
logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[lowmem] WARNING: invalid {name}={value!r}; using {default}")
        return default


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _save_query_dict(config, ckpt_config, save_embed_path, query_dict):
    os.makedirs(save_embed_path, exist_ok=True)
    if config.data_config.enable_query_instruct:
        name = 'query_SFpretrained_instruction_IT_dict.pt' if ckpt_config.using_pretrained else 'query_CLIPpretrained_instruction_IT_dict.pt'
    else:
        name = 'query_SFpretrained_IT_dict.pt' if ckpt_config.using_pretrained else 'query_CLIPpretrained_IT_dict.pt'
    path = os.path.join(save_embed_path, name)
    print(f'[lowmem] save query dict: {path}')
    torch.save(query_dict, path)
    return path


def _save_pool_dict(ckpt_config, save_embed_path, pool_dict):
    os.makedirs(save_embed_path, exist_ok=True)
    name = 'pool_SFpretrained_IT_dict.pt' if ckpt_config.using_pretrained else 'pool_CLIPpretrained_IT_dict.pt'
    path = os.path.join(save_embed_path, name)
    print(f'[lowmem] save pool dict: {path}')
    torch.save(pool_dict, path)
    return path


def main(config):
    """Low-memory train-feature extraction.

    Differences from the original script:
    1. Query dataset/loader is built and freed before candidate-pool dataset/loader.
       The original script keeps both the global train candidate pool dictionary and the
       global train candidate dataset in memory during query extraction, which can trigger
       OS SIGKILL on machines with limited RAM.
    2. num_workers, batch_size and pin_memory can be overridden by environment variables:
       LOWMEM_NUM_WORKERS, LOWMEM_BATCH_SIZE, PIN_MEMORY.
    3. If SKIP_EXISTING_TRAIN_FEATURES=1, existing query/pool pt files are skipped.
    """
    is_distributed_mode = config.dist_config.distributed_mode

    seed = config.seed + utils.get_rank()
    set_seed(seed)

    gpu_id = config.dist_config.gpu_id
    cudnn.benchmark = True

    # Low-memory overrides.
    config.dataloader_config.num_workers = _env_int('LOWMEM_NUM_WORKERS', int(config.dataloader_config.num_workers))
    config.dataloader_config.train_batch_size = _env_int('LOWMEM_BATCH_SIZE', int(config.dataloader_config.train_batch_size))
    pin_memory = _env_bool('PIN_MEMORY', False)
    skip_existing = _env_bool('SKIP_EXISTING_TRAIN_FEATURES', True)

    if utils.is_main_process():
        print('[lowmem] num_workers =', config.dataloader_config.num_workers)
        print('[lowmem] train_batch_size =', config.dataloader_config.train_batch_size)
        print('[lowmem] pin_memory =', pin_memory)
        print('[lowmem] skip_existing =', skip_existing)

    print("Creating CLIP model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")

    model = CLIPNoFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    model.float()

    ckpt_config = model_config.ckpt_config
    if ckpt_config.using_pretrained:
        checkpoint_path = os.path.join(config.genir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint["model"])

    model.eval()
    model = model.to(config.dist_config.gpu_id)

    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    save_embed_path = os.path.join(config.genir_dir, model_config.emb_save_path)
    os.makedirs(save_embed_path, exist_ok=True)
    query_out = os.path.join(
        save_embed_path,
        'query_SFpretrained_instruction_IT_dict.pt' if config.data_config.enable_query_instruct and ckpt_config.using_pretrained
        else 'query_SFpretrained_IT_dict.pt' if ckpt_config.using_pretrained
        else 'query_CLIPpretrained_instruction_IT_dict.pt' if config.data_config.enable_query_instruct
        else 'query_CLIPpretrained_IT_dict.pt',
    )
    pool_out = os.path.join(
        save_embed_path,
        'pool_SFpretrained_IT_dict.pt' if ckpt_config.using_pretrained else 'pool_CLIPpretrained_IT_dict.pt',
    )

    # ---------------- Query extraction ----------------
    if skip_existing and os.path.exists(query_out):
        if utils.is_main_process():
            print(f'[lowmem] query feature exists; skip query extraction: {query_out}')
    else:
        logger.info("Preparing query dataset ...")
        logger.info(f"Loading query dataset from {config.mbeir_data_dir}/{config.data_config.query_data_path} ...")
        query_dataset, query_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.QUERY,
        )
        query_sampler = DistributedSampler(
            dataset=query_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )
        query_loader = DataLoader(
            dataset=query_dataset,
            batch_size=config.dataloader_config.train_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=pin_memory,
            sampler=query_sampler,
            shuffle=False,
            collate_fn=query_collector,
            drop_last=False,
        )

        query_dict = {}
        query_id_to_index = {}
        query_img = []
        query_text = []
        query_img_mask = []
        query_text_mask = []

        if utils.is_main_process():
            print('Starting extraction query embedding!')
            q_index = 0

        for i, batch in enumerate(tqdm(query_loader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(gpu_id, non_blocking=True)

            qid_list = batch.get("qid_list")
            txt_batched = batch["txt_batched"]
            image_batched = batch["image_batched"]
            txt_mask_batched = batch["txt_mask_batched"]
            image_mask_batched = batch["image_mask_batched"]

            with torch.inference_mode():
                q_img_emb, q_txt_emb = model.module.encode_multimodal_input(image_batched, txt_batched)
            if not isinstance(qid_list, torch.Tensor):
                qid_list = torch.LongTensor(qid_list).to(gpu_id, non_blocking=True)

            q_text_mask = txt_mask_batched
            q_image_mask = image_mask_batched

            dist.barrier()
            if utils.get_world_size() > 1:
                qid_list = torch.cat(utils.GatherLayer.apply(qid_list), dim=0)
                q_txt_emb = torch.cat(utils.GatherLayer.apply(q_txt_emb), dim=0)
                q_img_emb = torch.cat(utils.GatherLayer.apply(q_img_emb), dim=0)
                q_text_mask = torch.cat(utils.GatherLayer.apply(txt_mask_batched), dim=0)
                q_image_mask = torch.cat(utils.GatherLayer.apply(image_mask_batched), dim=0)
            dist.barrier()

            if utils.is_main_process():
                for j, id in enumerate(qid_list):
                    id = id.item() if hasattr(id, "item") else int(id)
                    if id not in query_id_to_index:
                        query_id_to_index[id] = q_index
                        query_img.append(q_img_emb[j].detach().cpu())
                        query_text.append(q_txt_emb[j].detach().cpu())
                        query_img_mask.append(q_image_mask[j].detach().cpu())
                        query_text_mask.append(q_text_mask[j].detach().cpu())
                        q_index += 1

        dist.barrier()
        if utils.is_main_process():
            query_dict['img'] = torch.stack(query_img, dim=0)
            query_dict['text'] = torch.stack(query_text, dim=0)
            query_dict['img_mask'] = torch.stack(query_img_mask, dim=0)
            query_dict['text_mask'] = torch.stack(query_text_mask, dim=0)
            query_dict['id_to_index'] = query_id_to_index
            _save_query_dict(config, ckpt_config, save_embed_path, query_dict)

        # Free global cand-pool dictionary held by the query dataset before building cand dataset.
        del query_loader, query_sampler, query_dataset
        if utils.is_main_process():
            del query_dict, query_id_to_index, query_img, query_text, query_img_mask, query_text_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist.barrier()

    # ---------------- Candidate-pool extraction ----------------
    if skip_existing and os.path.exists(pool_out):
        if utils.is_main_process():
            print(f'[lowmem] pool feature exists; skip pool extraction: {pool_out}')
    else:
        logger.info("Preparing candidate-pool dataset ...")
        cand_dataset, cand_collector = build_mbeir_dataset_from_config(
            config=config,
            tokenizer=tokenizer,
            img_preprocess_fn=img_preprocess_fn,
            dataset_type=DatasetType.CAND,
        )
        cand_sampler = DistributedSampler(
            dataset=cand_dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=False,
        )
        cand_loader = DataLoader(
            dataset=cand_dataset,
            batch_size=config.dataloader_config.train_batch_size,
            num_workers=config.dataloader_config.num_workers,
            pin_memory=pin_memory,
            sampler=cand_sampler,
            shuffle=False,
            collate_fn=cand_collector,
            drop_last=False,
        )

        pool_dict = {}
        pool_id_to_index = {}
        pool_img = []
        pool_text = []
        pool_img_mask = []
        pool_text_mask = []

        if utils.is_main_process():
            print('Starting extraction pool embedding!')
            p_index = 0

        for i, batch in enumerate(tqdm(cand_loader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(gpu_id, non_blocking=True)

            did_list = batch.get("did_list")
            txt_batched = batch["txt_batched"]
            image_batched = batch["image_batched"]
            txt_mask_batched = batch["txt_mask_batched"]
            image_mask_batched = batch["image_mask_batched"]

            with torch.inference_mode():
                p_img_emb, p_txt_emb = model.module.encode_multimodal_input(image_batched, txt_batched)
            did_list = torch.LongTensor(did_list).to(p_txt_emb.device)

            p_image_mask = image_mask_batched
            p_text_mask = txt_mask_batched

            dist.barrier()
            if utils.get_world_size() > 1:
                did_list = torch.cat(utils.GatherLayer.apply(did_list), dim=0)
                p_txt_emb = torch.cat(utils.GatherLayer.apply(p_txt_emb), dim=0)
                p_img_emb = torch.cat(utils.GatherLayer.apply(p_img_emb), dim=0)
                p_image_mask = torch.cat(utils.GatherLayer.apply(image_mask_batched), dim=0)
                p_text_mask = torch.cat(utils.GatherLayer.apply(txt_mask_batched), dim=0)
            dist.barrier()

            if utils.is_main_process():
                for j, id in enumerate(did_list):
                    id = id.item() if hasattr(id, "item") else int(id)
                    if id not in pool_id_to_index:
                        pool_id_to_index[id] = p_index
                        pool_img.append(p_img_emb[j].detach().cpu())
                        pool_text.append(p_txt_emb[j].detach().cpu())
                        pool_img_mask.append(p_image_mask[j].detach().cpu())
                        pool_text_mask.append(p_text_mask[j].detach().cpu())
                        p_index += 1

        dist.barrier()
        if utils.is_main_process():
            pool_dict['img'] = torch.stack(pool_img, dim=0)
            pool_dict['text'] = torch.stack(pool_text, dim=0)
            pool_dict['img_mask'] = torch.stack(pool_img_mask, dim=0)
            pool_dict['text_mask'] = torch.stack(pool_text_mask, dim=0)
            pool_dict['id_to_index'] = pool_id_to_index
            _save_pool_dict(ckpt_config, save_embed_path, pool_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--genir_dir",
        type=str,
        default="/data/GENIUS",
        help="Path to GENIUS directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/GENIUS/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    # Parse arguments to config
    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    # Initialize distributed training
    args.dist_url = config.dist_config.dist_url  # Note: The use of args is a historical artifact :(
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    main(config)

    # Destroy the process group
    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()
