import torch
from torch.cuda.amp import autocast
from models.uniir_clip import utils




def _soundstorm_use_amp(config):
    trainer_cfg = getattr(config, "trainer_config", None)
    use_amp = bool(getattr(trainer_cfg, "use_amp", True))
    if bool(getattr(trainer_cfg, "disable_amp_for_soundstorm", False)):
        return False
    return use_amp

def _uses_soundstorm_scheduler(config):
    return str(getattr(config.trainer_config, "scheduler", "")).lower() in {
        "soundstorm", "soundstorm_reduce_on_plateau_warmup", "reduce_on_plateau_warmup"
    }


def _maybe_clip_gradients(model, config, scheduler):
    trainer_cfg = config.trainer_config
    clip_source = str(getattr(trainer_cfg, "clip_grad_norm_source", "default")).lower()
    if clip_source == "soundstorm":
        step = int(getattr(scheduler, "last_epoch", 0))
        start = int(getattr(trainer_cfg, "clip_grad_norm_start_iteration", 0))
        end = int(getattr(trainer_cfg, "clip_grad_norm_end_iteration", 5000))
        if start <= step <= end:
            max_norm = float(getattr(trainer_cfg, "clip_grad_norm_max_norm", 0.5))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return
    max_norm = float(getattr(trainer_cfg, "clip_grad_norm_max_norm", 1.0))
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _step_scheduler(scheduler, loss_value, config):
    if _uses_soundstorm_scheduler(config):
        scheduler.step(loss_value)
    else:
        scheduler.step()

def train_one_epoch(model, clip_model, data_loader, optimizer, epoch, gpu_id, scheduler, global_step, scaler, config):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("R_at_1", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = config.trainer_config.print_freq

    accumulation_steps = config.trainer_config.gradient_accumulation_steps
    accumulation_counter = 0

    model.train()
    if hasattr(model, 'module'):
        model.module.quantizer.eval()
    else:
        model.quantizer.eval()

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        max_train_batches = int(getattr(config.trainer_config, "max_train_batches", 0) or 0)
        if max_train_batches > 0 and i >= max_train_batches:
            break
        # autocast for mixed precision
        with autocast(enabled=_soundstorm_use_amp(config)):
            outputs = model(batch)
            loss = outputs["loss"]
            _bad_keys = []
            for _k, _v in outputs.items():
                if torch.is_tensor(_v) and _v.numel() == 1 and not torch.isfinite(_v.detach()).all():
                    _bad_keys.append(_k)
            if _bad_keys:
                print(f"[soundstorm-nan-guard] skip batch epoch={epoch} iter={i}: non-finite outputs={_bad_keys}")
                optimizer.zero_grad(set_to_none=True)
                for param in model.parameters():
                    param.grad = None
                continue

        # Scale the loss by the number of accumulation steps since backward
        # accumulates gradients.  Unscale/clip only immediately before an
        # optimizer step; calling unscale_ repeatedly within one accumulation
        # window is invalid for GradScaler.
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        accumulation_counter += 1
        if accumulation_counter == accumulation_steps:
            global_step += 1

            scaler.unscale_(optimizer)
            _bad_grad = False
            for _p in model.parameters():
                if _p.grad is not None and not torch.isfinite(_p.grad).all():
                    _bad_grad = True
                    break
            if _bad_grad:
                print(f"[soundstorm-nan-guard] skip optimizer step epoch={epoch} iter={i}: non-finite gradients")
                optimizer.zero_grad(set_to_none=True)
                for param in model.parameters():
                    param.grad = None
                scaler.update()
                accumulation_counter = 0
                continue
            _maybe_clip_gradients(model, config, scheduler)

            # optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            for param in model.parameters():
                param.grad = None
            _step_scheduler(scheduler, float(loss.detach().item()), config)
            accumulation_counter = 0

        metric_logger.update(R_at_1=outputs["R_at_1"].item())  # We scale back the loss for logging.
        metric_logger.update(loss=outputs["loss"].item())
        if "hdgr_mask_rate" in outputs:
            metric_logger.update(hdgr_mask_rate=outputs["hdgr_mask_rate"].item())
        hierarchy_metrics = (
            "token_acc", "masked_token_acc", "masked_seq_acc", "code_exact_acc",
            "modality_acc",
            "coarse_token_acc", "coarse_block_acc",
            "middle_token_acc", "middle_block_acc",
            "fine_token_acc", "fine_block_acc",
            "Level1_acc", "Level12_acc", "Level123_acc",
            "denoise_loss", "prefix_suffix_rate", "query_mix_rate",
            "residual_composition_scale", "residual_route_loss", "residual_branch_loss",
            "residual_regret_loss", "residual_no_harm_loss", "residual_clear_ce_loss",
            "residual_route_acc", "residual_clear_fraction", "residual_positive_gain_fraction",
            "residual_branch_acc", "residual_baseline_branch_acc",
            "residual_strength_mean", "residual_strength_std",
            "residual_oracle_strength_mean", "residual_expert0_probability",
            "residual_policy_entropy", "residual_utility_gain",
            "residual_best_utility_gain", "residual_utility_gap",
            "residual_expert0_win_rate", "residual_expert10_win_rate",
            "residual_expert50_win_rate", "residual_expert100_win_rate",
            "residual_composition_count", "residual_clear_count",
            "tree_risk_scale", "tree_branch_loss", "tree_branch_acc",
            "tree_branch_positive_mass", "tree_branch_count",
            "tree_prefix_risk_loss", "tree_prefix_risk_acc",
            "tree_prefix_risk_margin", "tree_prefix_risk_count",
            "soundstorm_t_mean", "soundstorm_changed_rate", "soundstorm_random_rate",
            "one_pass_rank_loss", "one_pass_rank_acc", "one_pass_hard_margin",
            "one_pass_full_mask_rate",
        )
        for optional_metric in hierarchy_metrics:
            if optional_metric in outputs:
                metric_logger.update(**{optional_metric: outputs[optional_metric].item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  # TODO: might need to loop through all param groups

    # Flush a final partial accumulation window.  Most current configs use one
    # step, but this keeps larger accumulation settings mathematically correct.
    if accumulation_counter > 0:
        scaler.unscale_(optimizer)
        correction = float(accumulation_steps) / float(accumulation_counter)
        if correction != 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(correction)
        _maybe_clip_gradients(model, config, scheduler)
        scaler.step(optimizer)
        scaler.update()
        for param in model.parameters():
            param.grad = None
        _step_scheduler(scheduler, float(loss.detach().item()), config)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if getattr(meter, "count", 0) > 0}


@torch.no_grad()
def eval_engine(model, clip_model, data_loader, gpu_id, config):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("R_at_1", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("lm_loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = "Test:"
    print_freq = config.evaluator.print_freq

    model.eval()
    num_samples = 0
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(gpu_id, non_blocking=True)  # Batch is a dictionary of tensors

        num_samples += len(batch)

        # autocast for mixed precision
        with autocast(enabled=_soundstorm_use_amp(config)):
            outputs = model(batch)
            loss = outputs["loss"]


        metric_logger.update(R_at_1=outputs["R_at_1"].item())  # We scale back the loss for logging.
        metric_logger.update(lm_loss=outputs["lm_loss"].item())  # We scale back the loss for logging.
        hierarchy_metrics = (
            "token_acc", "masked_token_acc", "masked_seq_acc", "code_exact_acc",
            "modality_acc",
            "coarse_token_acc", "coarse_block_acc",
            "middle_token_acc", "middle_block_acc",
            "fine_token_acc", "fine_block_acc",
            "denoise_loss", "prefix_suffix_rate", "query_mix_rate",
            "residual_composition_scale", "residual_route_loss", "residual_branch_loss",
            "residual_regret_loss", "residual_no_harm_loss", "residual_clear_ce_loss",
            "residual_route_acc", "residual_clear_fraction", "residual_positive_gain_fraction",
            "residual_branch_acc", "residual_baseline_branch_acc",
            "residual_strength_mean", "residual_strength_std",
            "residual_oracle_strength_mean", "residual_expert0_probability",
            "residual_policy_entropy", "residual_utility_gain",
            "residual_best_utility_gain", "residual_utility_gap",
            "residual_expert0_win_rate", "residual_expert10_win_rate",
            "residual_expert50_win_rate", "residual_expert100_win_rate",
            "residual_composition_count", "residual_clear_count",
            "tree_risk_scale", "tree_branch_loss", "tree_branch_acc",
            "tree_branch_positive_mass", "tree_branch_count",
            "tree_prefix_risk_loss", "tree_prefix_risk_acc",
            "tree_prefix_risk_margin", "tree_prefix_risk_count",
            "soundstorm_t_mean", "soundstorm_changed_rate", "soundstorm_random_rate",
        )
        for optional_metric in hierarchy_metrics:
            if optional_metric in outputs:
                metric_logger.update(**{optional_metric: outputs[optional_metric].item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if getattr(meter, "count", 0) > 0}
