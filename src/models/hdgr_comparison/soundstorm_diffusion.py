"""SoundStorm / VQ-Diffusion style discrete diffusion objective for GPT-HDGR.

This module intentionally follows the SoundStorm implementation uploaded by the
user, with only interface-level adaptation:
  - target acoustic tokens -> retrieval semantic-ID tokens [B, L]
  - SoundStorm transformer(cond, xt, t) -> existing GPT-HDGR/HDGR id_generator
  - condition dictionary -> query-conditioned prefix embeddings

The formulas are copied/adapted from:
  SoundStorm-master/soundstorm/s2/models/dalle_wav/diffusion_transformer.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EPS = 1.0e-8


def sum_except_batch(x: torch.Tensor, num_dims: int = 1) -> torch.Tensor:
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    return torch.log(1 - a.exp() + 1.0e-40)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    log_x_start = torch.nan_to_num(log_x_start.float(), nan=-70.0, posinf=0.0, neginf=-70.0).clamp(-70.0, 0.0)
    log_prob = torch.nan_to_num(log_prob.float(), nan=-70.0, posinf=0.0, neginf=-70.0).clamp(-70.0, 0.0)
    out = (log_x_start.exp() * log_prob).sum(dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
def index_to_log_onehot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"
    x_onehot = F.one_hot(x.long(), num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1.0e-30))
    return log_x


def log_onehot_to_index(log_x: torch.Tensor) -> torch.Tensor:
    return log_x.argmax(1)


def alpha_schedule(
    time_step: int,
    N: int = 100,
    att_1: float = 0.99999,
    att_T: float = 0.000009,
    ctt_1: float = 0.000009,
    ctt_T: float = 0.9,
):
    """SoundStorm/VQ-Diffusion alpha1 schedule.

    att: cumulative keep probability
    btt: cumulative uniform/random-token probability per real class
    ctt: cumulative mask probability
    """
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct

    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class SoundStormTokenDiffusion(nn.Module):
    """Discrete categorical diffusion objective copied from SoundStorm.

    The wrapped denoiser is supplied by ``model_forward_fn`` at call time. It must
    return real-vocab logits with shape [B, L, V], where V = num_classes - 1.
    The final class ``num_classes - 1`` is the mask token and is never predicted
    as x0, matching SoundStorm's zero-vector append in ``predict_start``.
    """

    def __init__(
        self,
        num_classes: int,
        num_timesteps: int = 100,
        alpha_init_type: str = "alpha1",
        auxiliary_loss_weight: float = 5.0e-4,
        adaptive_auxiliary_loss: bool = True,
        mask_weight=(1.0, 1.0),
        importance_min_count: int = 10,
        logit_min: float = -70.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_timesteps = int(num_timesteps)
        self.loss_type = "vb_stochastic"
        self.parametrization = "x0"
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        self.adaptive_auxiliary_loss = bool(adaptive_auxiliary_loss)
        self.mask_weight = tuple(float(x) for x in mask_weight)
        self.importance_min_count = int(importance_min_count)
        self.logit_min = float(logit_min)
        self.mask_token_id = self.num_classes - 1

        if alpha_init_type != "alpha1":
            raise ValueError("Only SoundStorm alpha_init_type='alpha1' is supported for the copied objective.")
        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes)

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))

        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.0e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.0e-5

        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())
        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

    def multinomial_kl(self, log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
        log_prob1 = torch.nan_to_num(log_prob1.float(), nan=self.logit_min, posinf=0.0, neginf=self.logit_min).clamp(self.logit_min, 0.0)
        log_prob2 = torch.nan_to_num(log_prob2.float(), nan=self.logit_min, posinf=0.0, neginf=self.logit_min).clamp(self.logit_min, 0.0)
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        kl = torch.nan_to_num(kl, nan=0.0, posinf=100.0, neginf=0.0)
        return kl.clamp_min(0.0).clamp_max(100.0)
    def q_pred_one_timestep(self, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_at = extract(self.log_at, t, log_x_t.shape)
        log_bt = extract(self.log_bt, t, log_x_t.shape)
        log_ct = extract(self.log_ct, t, log_x_t.shape)
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct),
            ],
            dim=1,
        )
        return log_probs

    def q_pred(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct),
            ],
            dim=1,
        )
        return log_probs

    def q_posterior(self, log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size(0)
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_token_id).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1, device=log_x_t.device, dtype=log_x_t.dtype)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_start.shape[2])

        log_qt = self.q_pred(log_x_t, t)
        log_qt = torch.cat((log_qt[:, :-1, :], log_zero_vector), dim=1)
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start - log_qt
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        return torch.nan_to_num(torch.clamp(log_EV_xtmin_given_xt_given_xstart, self.logit_min, 0.0), nan=self.logit_min, posinf=0.0, neginf=self.logit_min)

    def log_sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        logits_f = torch.nan_to_num(
            logits_f,
            nan=-1.0e9,
            posinf=1.0e9,
            neginf=-1.0e9,
        )
        logits_f = logits_f.clamp(min=-1.0e9, max=1.0e9)

        # Stable Gumbel-max categorical sampling over class dim=1.
        gumbel = -torch.empty_like(logits_f).exponential_().log()
        sample = torch.argmax(logits_f + gumbel, dim=1).long()
        sample = sample.clamp_(min=0, max=self.num_classes - 1)
        return index_to_log_onehot(sample, self.num_classes)
    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        return self.log_sample_categorical(log_EV_qxt_x0)

    def sample_time(self, b: int, device: torch.device, method: str = "importance"):
        if method == "importance":
            try:
                ready = bool((self.Lt_count > self.importance_min_count).all().item())
            except Exception:
                ready = False

            if not ready:
                return self.sample_time(b, device, method="uniform")

            pt_all = torch.sqrt(torch.clamp(self.Lt_history.detach().float(), min=0.0) + 1e-10) + 1e-4
            pt_all = pt_all.to(device)

            if pt_all.numel() >= 2:
                pt_all = pt_all.clone()
                pt_all[0] = pt_all[1]

            pt_all = torch.nan_to_num(pt_all, nan=0.0, posinf=0.0, neginf=0.0)
            pt_all = torch.clamp(pt_all, min=0.0)
            s = pt_all.sum()
            if (not bool(torch.isfinite(s).item())) or float(s.item()) <= 0.0:
                pt_all = torch.ones(int(self.num_timesteps), device=device, dtype=torch.float32) / float(self.num_timesteps)
            else:
                pt_all = pt_all / s.clamp_min(1e-20)

            # Stable Gumbel-max timestep sampling.  Avoid torch.multinomial CUDA assert.
            log_pt = torch.log(pt_all.clamp_min(1e-20))
            gumbel = -torch.empty((b, int(self.num_timesteps)), device=device, dtype=log_pt.dtype).exponential_().log()
            t = torch.argmax(log_pt.unsqueeze(0) + gumbel, dim=1).long()
            pt = pt_all.gather(0, t).clamp_min(1e-4)
            return t, pt

        if method == "uniform":
            t = torch.randint(0, int(self.num_timesteps), (b,), device=device).long()
            pt = torch.ones((b,), device=device, dtype=torch.float32) / float(self.num_timesteps)
            return t, pt

        raise ValueError(f"Unknown sample_time method: {method}")
    def corruption_stats(self, x_start: torch.Tensor, x_t: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = valid_mask.bool()
        valid_count = valid.sum().clamp_min(1)
        mask_region = (x_t == self.mask_token_id) & valid
        changed_region = (x_t != x_start) & valid
        random_region = changed_region & (~mask_region)
        return {
            "soundstorm_mask_rate": mask_region.sum().float() / valid_count.float(),
            "soundstorm_changed_rate": changed_region.sum().float() / valid_count.float(),
            "soundstorm_random_rate": random_region.sum().float() / valid_count.float(),
        }

    def predict_start_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert real-vocab logits [B,L,V] to SoundStorm log_x0 [B,V+1,L]."""
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        log_pred = F.log_softmax(logits.transpose(1, 2), dim=1)
        log_pred = torch.nan_to_num(log_pred, nan=self.logit_min, posinf=0.0, neginf=self.logit_min).clamp(self.logit_min, 0.0)
        batch_size = logits.size(0)
        zero_vector = torch.zeros(batch_size, 1, logits.size(1), device=logits.device, dtype=log_pred.dtype) + self.logit_min
        log_pred = torch.cat((log_pred, zero_vector), dim=1)
        return torch.clamp(log_pred, self.logit_min, 0.0)
    def train_loss(
        self,
        x_start: torch.Tensor,
        valid_mask: torch.Tensor,
        model_forward_fn,
        is_train: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the copied SoundStorm training objective.

        Args:
            x_start: [B, L] clean semantic-ID token ids in [0, V-1].
            valid_mask: [B, L], true for real target tokens.
            model_forward_fn: callable(xt_ids, t) -> logits [B, L, V].
        """
        b, device = x_start.size(0), x_start.device
        valid = valid_mask.bool()
        t, pt = self.sample_time(b, device, "importance")
        pt = torch.nan_to_num(pt.float(), nan=1.0 / float(self.num_timesteps), posinf=1.0, neginf=1.0 / float(self.num_timesteps)).clamp_min(1.0e-4)

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        logits = model_forward_fn(xt, t)
        log_x0_recon = self.predict_start_from_logits(logits)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)

        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)

        mask_region = (xt == self.mask_token_id).float()
        mask_weight = mask_region * self.mask_weight[0] + (1.0 - mask_region) * self.mask_weight[1]
        valid_float = valid.float()
        kl = kl * mask_weight * valid_float
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = decoder_nll * valid_float
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1.0 - mask) * kl
        kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=1.0e4, neginf=0.0).clamp(0.0, 1.0e4)

        # Copy SoundStorm's Lt history update for importance sampling.
        with torch.no_grad():
            Lt2 = torch.nan_to_num(kl_loss.pow(2), nan=0.0, posinf=1.0e8, neginf=0.0).clamp(0.0, 1.0e8)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        vb_loss = kl_loss / pt
        vb_loss = torch.nan_to_num(vb_loss, nan=0.0, posinf=1.0e4, neginf=0.0).clamp(0.0, 1.0e4)

        if self.auxiliary_loss_weight != 0 and is_train:
            kl_aux = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon[:, :-1, :])
            kl_aux = kl_aux * mask_weight * valid_float
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1.0 - mask) * kl_aux
            kl_aux_loss = torch.nan_to_num(kl_aux_loss, nan=0.0, posinf=1.0e4, neginf=0.0).clamp(0.0, 1.0e4)
            if self.adaptive_auxiliary_loss:
                addition_loss_weight = t.float() / float(self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0
            vb_loss = vb_loss + addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss = torch.nan_to_num(vb_loss, nan=0.0, posinf=1.0e4, neginf=0.0).clamp(0.0, 1.0e4)

        valid_count = valid_float.sum().clamp_min(1.0)
        loss = vb_loss.sum() / valid_count
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0e4, neginf=0.0)

        pred_token_ids = log_x0_recon[:, :-1, :].argmax(dim=1)
        token_correct = pred_token_ids == x_start
        seq_acc = (token_correct | ~valid).all(dim=1).float().mean()
        token_acc = (token_correct & valid).sum().float() / valid_count
        level1_acc = (pred_token_ids[:, :1] == x_start[:, :1]).all(dim=1).float().mean()
        level12_acc = (pred_token_ids[:, :min(2, x_start.size(1))] == x_start[:, :min(2, x_start.size(1))]).all(dim=1).float().mean()
        level123_acc = (pred_token_ids[:, :min(3, x_start.size(1))] == x_start[:, :min(3, x_start.size(1))]).all(dim=1).float().mean()

        mask_pos = (xt == self.mask_token_id) & valid
        masked_count = mask_pos.sum().clamp_min(1)
        masked_token_acc = (token_correct & mask_pos).sum().float() / masked_count.float()
        masked_seq_acc = ((token_correct | ~mask_pos).all(dim=1)).float().mean()

        stats = self.corruption_stats(x_start, xt, valid_mask)
        return {
            "loss": loss,
            "vb_loss": loss.detach(),
            "logits": logits,
            "log_x0_recon": log_x0_recon,
            "xt": xt,
            "t": t,
            "pred_token_ids": pred_token_ids,
            "seq_acc": seq_acc,
            "token_acc": token_acc,
            "masked_token_acc": masked_token_acc,
            "masked_seq_acc": masked_seq_acc,
            "level1_acc": level1_acc,
            "level12_acc": level12_acc,
            "level123_acc": level123_acc,
            "t_mean": t.float().mean() / max(1.0, float(self.num_timesteps - 1)),
            **stats,
        }
