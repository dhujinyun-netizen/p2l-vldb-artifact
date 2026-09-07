"""
Official GENIUS autoregressive generator baseline.

This is adapted from GENIUS `retriever_oral.py` and kept intentionally
separate from HDGR block-denoising and GPT-diffusion baselines.  The model
uses the official T5-style AR generator: T5ForConditionalGeneration with
zero encoder layers, learned/query prefix embeddings, RQ-codebook token
initialization, and Trie-constrained beam search.

Local adaptations:
- supports the current HDGR dataset batch format `(query, pool, h_qid)` as
  well as the legacy GENIUS `(query, pool, instruct, h_qid)` format;
- can build its code-token tokenizer without downloading a T5 tokenizer;
- avoids torch.compile during inference unless explicitly enabled.
"""

# Standard library
import os
import re
import math
import random
import string
from typing import Optional, Tuple, List, Dict, Any
import pickle

# Third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from einops import rearrange
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Config,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# Local modules
from models.uniir_clip import utils
from models.generative_retriever.beam import Trie, MarisaTrie
try:
    import models.generative_retriever.trie_cpp as Trie_Cpp
except ModuleNotFoundError:
    Trie_Cpp = None
from models.residual_quantization.residual_quantization import RQ
from models.residual_quantization.loss import ClipLoss

# ================ Constants ================
IGNORE_INDEX = -100


def cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _make_t5_small_config_offline():
    """Create a T5-small-like config without requiring HF network/cache."""
    cfg = T5Config(
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=0,
        num_decoder_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
    )
    return cfg

# ================ Loss Functions ================

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.01, metric: str = 'cos', bidirection: bool = True, gather: bool = True):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        self.gather = gather
        self.bidirection = bidirection

    def get_ground_truth(self, device: torch.device, num_logits: int) -> torch.Tensor:
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def forward(self, x: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None, logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logit is None:
            if utils.get_world_size() > 1 and self.gather:
                x = torch.cat(utils.GatherLayer.apply(x), dim=0)
                y = torch.cat(utils.GatherLayer.apply(y), dim=0)

            assert x is not None and y is not None
            if self.metric == 'cos':
                logits_per_x = F.linear(F.normalize(x), F.normalize(y))
            elif self.metric == 'euclid':
                logits_per_x = -torch.cdist(x,y) ** 2
            else:
                raise ValueError(f'Invalid metric: {self.metric}')
            labels = self.get_ground_truth(x.device, x.shape[0])

        else:
            logits_per_x = logit
            labels = self.get_ground_truth(logit.device, logit.shape[0])

        logits_per_x = logits_per_x / self.temperature
        logits_per_y = logits_per_x.T

        if self.bidirection:
            total_loss = (F.cross_entropy(logits_per_x, labels) + F.cross_entropy(logits_per_y, labels)) / 2
        else:
            total_loss = F.cross_entropy(logits_per_x, labels)
        return total_loss

# ================ Main Model ================

class T5ForGenerativeRetrieval(nn.Module):
    def __init__(self, config=None, tokenizer=None, clip_model=None, new_tokenizer=True, init_rq_codebook=True):
        super().__init__()

        if clip_model is not None:
            self.clip_model = clip_model
            for _, param in self.clip_model.named_parameters():
                param.requires_grad = False
            self.clip_model.eval()

        self.config = config
        self.quantizer = RQ(config=config, clip_model=clip_model)
        rq_model_path = os.path.join(config.genir_dir, config.codebook_config.quantizer_path)

        # 加载权重：显式关闭 weights_only 以支持旧版 checkpoint 包含的 DictConfig
        self.quantizer.load_state_dict(
            torch.load(rq_model_path, map_location=torch.device('cpu'), weights_only=False)["model"], strict=False)
        self.quantizer.eval()
        for _, param in self.quantizer.named_parameters():
            param.requires_grad = False
        self.modality_index = self.quantizer.modality_index

        # Official GENIUS AR uses a T5-style seq2seq model with no encoder layers.
        # Prefer a local/cached config when available, but never require network.
        t5_name = cfg_get(cfg_get(config, "model", {}), "t5_model_name", "google-t5/t5-small")
        try:
            t5_config = AutoModelForSeq2SeqLM.from_pretrained(t5_name, local_files_only=True).config
            t5_config.num_layers = 0
            if getattr(t5_config, "num_decoder_layers", None) is None:
                t5_config.num_decoder_layers = 6
        except Exception as exc:
            print(f"[GENIUS_AR] Could not load local T5 config {t5_name!r}; using offline T5-small config. Reason: {exc}")
            t5_config = _make_t5_small_config_offline()
        self.id_generator = T5ForConditionalGeneration(t5_config)
        self.id_generator.config.decoder_start_token_id = 0

        self.num_prefix = int(cfg_get(cfg_get(config, "model", {}), "num_prefix", 30))
        self.embed_projector = nn.Sequential(nn.Linear(768, t5_config.d_model * self.num_prefix))
        self.embed_projector.train()
        for _, param in self.embed_projector.named_parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0)
        self.codebook_vocab = self.quantizer.codebook_vocab
        self.codebook_level = self.quantizer.codebook_level
        self.iter = 0
        self.alpha = getattr(getattr(config, 'hyperparameter_config', {}), 'alpha', 2)
        self.tokenizer = tokenizer
        self.trie_index = None
        self.trie_type = getattr(getattr(config, 'model', {}), 'trie_type', 'trie_cpp')
        self._compiled_id_gen = False
        self.code_tokens = []

        if self.quantizer.unique_code:
            self.codebook_level = self.codebook_level + 1

        if new_tokenizer:
            self._initialize_tokenizer(tokenizer)

        self._initialize_codebook_tokens()

        if init_rq_codebook:
            self._initialize_codebook_embeddings(t5_config)

        print(f"[GENIUS_AR] Initialized official autoregressive GENIUS generator | code_length={self.codebook_level} vocab={len(self.tokenizer)} prefix={self.num_prefix}")

    def _initialize_tokenizer(self, tokenizer):
        special_tokens = {'pad_token': '<pad>', 'eos_token': '</s>', 'unk_token': '<unk>'}
        if tokenizer is not None:
            try:
                special_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens.values()]
                new_vocab = {tokenizer.convert_ids_to_tokens(idx): idx for idx in special_token_ids}
            except Exception:
                new_vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2}
        else:
            new_vocab = {'<pad>': 0, '</s>': 1, '<unk>': 2}
        tokenizer_model = WordLevel(vocab=new_vocab, unk_token='<unk>')
        tokenizer = Tokenizer(tokenizer_model)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens)
        self.tokenizer = tokenizer
        self.id_generator.resize_token_embeddings(len(tokenizer))

    def _initialize_codebook_tokens(self):
        self.level_indicators = list(string.ascii_lowercase[:self.codebook_level])
        for l, level in enumerate(self.level_indicators):
            if self.modality_index and l == 0:
                for i in range(3):
                    self.code_tokens.append(f'<{level}{i}>')
                continue
            for i in range(self.codebook_vocab):
                self.code_tokens.append(f'<{level}{i}>')
        num_new_tokens = self.tokenizer.add_tokens(self.code_tokens)
        self.id_generator.resize_token_embeddings(len(self.tokenizer))

    def _initialize_codebook_embeddings(self, t5_config):
        new_token_ids = self.tokenizer.convert_tokens_to_ids(self.code_tokens)
        embedding_dim = t5_config.d_model
        linear_layer = nn.Linear(768, embedding_dim, bias=False)

        if self.modality_index:
            first_layer_codebook = self.quantizer.residual_rq.layers[0]._codebook.embed[0, :3, :]
            mapped_first_layer_embeddings = linear_layer(F.normalize(first_layer_codebook))

            other_layers_codebooks = [layer._codebook.embed for layer in self.quantizer.residual_rq.layers[1:]]
            other_layers_codebooks = torch.stack(other_layers_codebooks, dim=0)
            other_layers_codebooks = rearrange(other_layers_codebooks, 'q 1 c d -> q c d')
            other_layers_codebook = other_layers_codebooks.reshape(-1, other_layers_codebooks.shape[-1])
            mapped_other_layers_embeddings = linear_layer(F.normalize(other_layers_codebook))

            mapped_embeddings = torch.cat([mapped_first_layer_embeddings, mapped_other_layers_embeddings], dim=0)
        else:
            codebook_vectors = self.quantizer.residual_rq.codebooks.reshape(-1, 768)
            mapped_embeddings = linear_layer(F.normalize(codebook_vectors))

        self._initialize_model_embeddings(new_token_ids, mapped_embeddings)

    def _initialize_model_embeddings(self, new_token_ids, mapped_embeddings):
        with torch.no_grad():
            lm_head = self.id_generator.lm_head
            for idx, token_id in enumerate(new_token_ids):
                lm_head.weight[token_id] = mapped_embeddings[idx] * 100
        for param in lm_head.parameters():
            param.requires_grad = True

        with torch.no_grad():
            shared = self.id_generator.shared
            for idx, token_id in enumerate(new_token_ids):
                shared.weight[token_id] = mapped_embeddings[idx] * 100
        for param in shared.parameters():
            param.requires_grad = True
        self.id_generator._tie_weights()

    def transform_row(self, row, separator=''):
        transformed_row = []
        for l, level_indicator in enumerate(self.level_indicators):
            new_value = row[l]
            transformed_row.append(f"<{level_indicator}{new_value}>")
        return separator.join(transformed_row)

    def detransform_row(self, row):
        splited_row = re.findall(r'<.*?>', row)
        detransformed_row = []
        for token in splited_row:
            match = re.match(r'<([a-z])(\d+)>', token)
            if match:
                level_indicator, value = match.groups()
                detransformed_row.append(int(value))
        return detransformed_row

    def compute_single_batch(self, batch, gpu_id=None):
        """Compute loss and metrics for a single batch."""
        # 补丁：确保 gpu_id 始终为有效的 torch.device，防止非分布式模式下的 None 报错
        if not isinstance(gpu_id, torch.device):
            gpu_id = torch.device(f'cuda:{gpu_id}' if isinstance(gpu_id, int) else 'cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            query, pool, instruct, h_qid = batch
            if instruct is not None and torch.is_tensor(instruct):
                instruct = instruct.view(-1, instruct.size(-1)).to(gpu_id, non_blocking=True)
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            query, pool, h_qid = batch
            instruct = None
        else:
            raise ValueError(f"GENIUS_AR expected batch tuple/list of length 3 or 4, got {type(batch)} len={len(batch) if hasattr(batch, '__len__') else 'NA'}")

        h_qid = h_qid.view(-1)

        q_img_mask = query['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        q_txt_mask = query['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_img_mask = pool['img_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)
        p_txt_mask = pool['txt_mask'].view(-1).unsqueeze(-1).to(gpu_id, non_blocking=True)

        q_img_emb = query['img_emb'].view(-1, query['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        q_txt_emb = query['txt_emb'].view(-1, query['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_img_emb = pool['img_emb'].view(-1, pool['img_emb'].size(-1)).to(gpu_id, non_blocking=True)
        p_txt_emb = pool['txt_emb'].view(-1, pool['txt_emb'].size(-1)).to(gpu_id, non_blocking=True)

        device = q_img_emb.device
        bs = len(q_img_emb)

        # 此处 self.quantizer.inference 涉及多模块运算，依赖 forward 里的 to(device) 保证同步
        q_output = self.quantizer.inference(q_img_emb, q_txt_emb, q_img_mask, q_txt_mask)
        q_emb, q_code_list = q_output['encode'], q_output['code'].tolist()
        p_output = self.quantizer.inference(p_img_emb, p_txt_emb, p_img_mask, p_txt_mask)
        p_emb, p_code_list = p_output['encode'], p_output['code'].tolist()
        q_emb, p_emb = F.normalize(q_emb), F.normalize(p_emb)

        p_code_str_list = [self.transform_row(row) for row in p_code_list]
        s = torch.distributions.Beta(self.alpha, self.alpha).sample((bs, q_emb.size(1))).to(device)
        aug_emb = torch.sqrt(s) * q_emb + torch.sqrt(1-s) * p_emb
        aug_proj = self.embed_projector(F.normalize(aug_emb)).to(q_emb.dtype)
        inputs_embeds = aug_proj.reshape(bs, self.num_prefix, -1)

        label_id = p_code_str_list
        labels = self.tokenizer(label_id, padding=True, truncation=True).input_ids
        labels = torch.LongTensor(labels).to(device)
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        model_outputs = self.id_generator(inputs_embeds=inputs_embeds,
                                        labels=labels,
                                        output_hidden_states=True)

        logits = model_outputs.logits
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        pred_idx = logits.argmax(-1)
        R_at_1 = (pred_idx[:bs] == labels[:bs]).all(1).sum() / bs
        Level1_acc = (pred_idx[:bs,1:2] == labels[:bs,1:2]).all(1).sum() / bs
        Level12_acc = (pred_idx[:bs,1:3] == labels[:bs,1:3]).all(1).sum() / bs
        Level123_acc = (pred_idx[:bs,1:4] == labels[:bs,1:4]).all(1).sum() / bs

        outputs = {
            'loss': loss,
            'R_at_1': R_at_1,
            'Level1_acc': Level1_acc,
            'Level12_acc': Level12_acc,
            'Level123_acc': Level123_acc
        }

        if self.iter % 200 == 0:
            example = self.tokenizer.decode(pred_idx[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            example_labels = self.tokenizer.decode(labels[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print('Retrieval Task: ' + 'Pred: ' + str(example) + ' Ans: ' + str(example_labels))

        self.iter += 1
        return outputs

    def get_img_preprocess_fn(self):
        return self.clip_model.get_img_preprocess_fn()

    def get_clip_tokenizer(self):
        return self.clip_model.get_tokenizer()

    def get_seq2seq_tokenizer(self):
        return self.tokenizer

    def generative_index(self, cand_codes):
        model_config = self.config.model
        ckpt_config = model_config.ckpt_config
        trie_path = os.path.join(self.config.genir_dir, ckpt_config.ckpt_dir)
        cand_codes_str = [self.transform_row(row) for row in cand_codes]
        cand_codes_ids = np.array(self.tokenizer(cand_codes_str, add_special_tokens=False).input_ids).tolist()

        if self.trie_type == 'marisa':
            trie = MarisaTrie(cand_codes_ids)
        elif self.trie_type == 'triecpp':
            trie = Trie_Cpp.Trie(cand_codes_ids)
        else:
            trie = Trie(cand_codes_ids)
        return trie

    def distribute_trie(self, cand_codes, trie_save_path):
        if trie_save_path.endswith("trie.pkl"):
            suffix_map = {
                "trie": "trie.pkl",
                "triecpp": "triecpp.pkl",
                "triemarisa": "triemarisa.pkl",
            }
            base_path = trie_save_path[:-len("trie.pkl")]
            trie_save_path = base_path + suffix_map.get(self.trie_type, "trie.pkl")

        if self.trie_type == 'triecpp':
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0 and not os.path.exists(trie_save_path):
                trie_index = self.generative_index(cand_codes)
                with open(trie_save_path, 'wb') as f:
                    pickle.dump(trie_index.to_dict(), f)
                    f.flush()
                    os.fsync(f.fileno())
                del trie_index
                print(f"Log: Save candidate pool Trie from {trie_save_path}.")

            if dist.is_initialized():
                dist.barrier()

            if hasattr(self, 'trie_index'):
                del self.trie_index
                torch.cuda.empty_cache()

            with open(trie_save_path, 'rb') as f:
                trie_dict = pickle.load(f)
            self.trie_index_dict = trie_dict
            self.trie_index = Trie_Cpp.Trie.from_dict(trie_dict)
            del trie_dict

        else:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0 and not os.path.exists(trie_save_path):
                trie_index = self.generative_index(cand_codes)
                with open(trie_save_path, 'wb') as f:
                    pickle.dump(trie_index, f)
                    f.flush()
                    os.fsync(f.fileno())
                del trie_index
                print(f"Log: Save candidate pool Trie from {trie_save_path}.")

            if dist.is_initialized():
                dist.barrier()

            if hasattr(self, 'trie_index'):
                del self.trie_index
                torch.cuda.empty_cache()

            with open(trie_save_path, 'rb') as f:
                self.trie_index = pickle.load(f)

        print(f"Log: Loaded candidate pool Trie from {trie_save_path}.")

    def constrained_beam_search(self, inputs_embeds, attention_mask=None, num_beams=10, cand_codes=None):
        batch_size = inputs_embeds.size(0)
        device = inputs_embeds.device

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            prefix = input_ids.tolist()
            if prefix[0] == self.tokenizer.pad_token_id:
                prefix = prefix[1:]
            valid_tokens = self.trie_index.get(prefix)
            if valid_tokens is None:
                return []
            else:
                return valid_tokens

        with torch.no_grad():
            generated = self.id_generator.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                use_cache=True,
                max_new_tokens=self.codebook_level,
                early_stopping=True,
            )
        return generated

    @torch.no_grad()
    def inference(self, img_emb, txt_emb, img_mask, txt_mask, inst_ids, num_beams=10, cand_codes=None):
        device = img_emb.device
        bs = len(img_emb)

        # Keep official AR behavior, but avoid torch.compile by default because it
        # increases first-eval latency and can be brittle across torch versions.
        if bool(cfg_get(cfg_get(self.config, "model", {}), "compile_id_generator", False)) and not self._compiled_id_gen:
            self.id_generator = torch.compile(self.id_generator)
            self._compiled_id_gen = True

        q_output = self.quantizer.inference(img_emb, txt_emb, img_mask, txt_mask)
        emb, q_code_list = q_output['encode'], q_output['code'].tolist()
        emb = F.normalize(emb)
        q_code_str_list = [self.transform_row(row) for row in q_code_list]

        inputs_embeds = self.embed_projector(emb).reshape(len(emb), self.num_prefix, -1)
        outputs = self.constrained_beam_search(inputs_embeds, num_beams=num_beams, cand_codes=cand_codes)

        outputs_id = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_id = [self.detransform_row(row) for row in outputs_id]
        outputs_id_tensor = torch.LongTensor(outputs_id).to(device)
        outputs_id_tensor = outputs_id_tensor.reshape(bs, num_beams, -1).detach()

        return outputs_id_tensor, F.normalize(img_emb * img_mask + txt_emb * txt_mask)

    def encode_mbeir_batch(self, batch, num_beams=10, cand_codes=None, trie_save_path=None, init_dataset=True):
        if init_dataset:
            self.distribute_trie(cand_codes, trie_save_path)

        id_list = batch.get("did_list") or batch.get("qid_list")
        assert id_list is not None, "id_list must be provided."
        assert isinstance(id_list[0], int), "id_list must be hashed to int."

        img_emb, txt_emb = self.clip_model.encode_multimodal_input(
            batch["image_batched"],
            batch["txt_batched"],
        )
        img_mask = batch["image_mask_batched"].unsqueeze(-1)
        txt_mask = batch["txt_mask_batched"].unsqueeze(-1)
        assert img_emb.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."

        output, embeddings = self.inference(
            img_emb,
            txt_emb,
            img_mask,
            txt_mask,
            batch.get("inst_ids", None),
            num_beams=num_beams,
            cand_codes=cand_codes
        )
        return output, embeddings, torch.LongTensor(id_list)

    def forward(self,
                input=None,
                encode_mbeir_batch=False,
                num_beams=10,
                cand_codes=None,
                init_dataset=True,
                trie_save_path=None,
                gpu_id=None):
        """
        Forward pass with Explicit Device Synchronization Patch.
        """
        # 1. 动态对齐目标设备
        if isinstance(gpu_id, torch.device):
            target_device = gpu_id
        elif isinstance(gpu_id, int):
            target_device = torch.device(f'cuda:{gpu_id}')
        else:
            target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 2. 强制全模型搬家
        self.to(target_device)

        # 3. 终极补丁：显式移动可能未被正确注册为子模块的组件
        if hasattr(self, 'quantizer'):
            self.quantizer.to(target_device)
        if hasattr(self, 'id_generator'):
            self.id_generator.to(target_device)
        if hasattr(self, 'embed_projector'):
            self.embed_projector.to(target_device)

        if encode_mbeir_batch:
            return self.encode_mbeir_batch(input, cand_codes=cand_codes, num_beams=num_beams, init_dataset=init_dataset,
                                           trie_save_path=trie_save_path)
        else:
            # 确保传递给下游的是转换后的 device 对象
            return self.compute_single_batch(input, target_device)