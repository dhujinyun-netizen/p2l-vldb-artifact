"""Factory for the two retained generators: GENIUS AR and StructNAR."""


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def generator_type(config):
    model_cfg = _cfg_get(config, "model", {})
    return str(_cfg_get(model_cfg, "generator_type", "")).lower()


def is_genius_ar_config(config) -> bool:
    gtype = generator_type(config)
    model_cfg = _cfg_get(config, "model", {})
    name = str(_cfg_get(model_cfg, "name", "")).lower()
    short_name = str(_cfg_get(model_cfg, "short_name", "")).lower()
    return (
        gtype in {"genius_ar", "official_genius_ar", "genius_autoregressive", "t5_ar"}
        or "genius_ar" in name
        or "genius_ar" in short_name
        or name.startswith("genius")
        or short_name.startswith("genius")
    )


def is_gpt_hdgr_config(config) -> bool:
    gtype = generator_type(config)
    model_cfg = _cfg_get(config, "model", {})
    name = str(_cfg_get(model_cfg, "name", "")).lower()
    short_name = str(_cfg_get(model_cfg, "short_name", "")).lower()
    backbone = str(_cfg_get(model_cfg, "hdgr_backbone", "")).lower()
    return (
        gtype in {"gpt_hdgr", "gpt_diffusion_hdgr", "gpt_hdgr_backbone", "gpt_with_hdgr"}
        or "gpt_hdgr" in name
        or "gpt_hdgr" in short_name
        or "gpt_hdgr" in backbone
    )


def is_hdgr_block_denoising_config(config) -> bool:
    model_cfg = _cfg_get(config, "model", {})
    gtype = generator_type(config)
    name = str(_cfg_get(model_cfg, "name", "")).lower()
    backbone = str(_cfg_get(model_cfg, "hdgr_backbone", _cfg_get(model_cfg, "bd3lm_backbone", ""))).lower()
    return (
        gtype in {"hdgr_block_denoising", "bd3lm_diffusion"}
        or "block_denoising" in gtype
        or "hdgr" in name
        or "block_denoising" in backbone
    )


def uses_t5_tokenizer(config) -> bool:
    """Whether to load a legacy T5 tokenizer before constructing the generator."""
    model_cfg = _cfg_get(config, "model", {})
    if not bool(_cfg_get(model_cfg, "use_t5_tokenizer", False)):
        return False
    # HDGR/GPT/GENIUS_AR baselines create/use code-token tokenizers internally.
    # GENIUS_AR follows the official T5-style generator architecture, but its
    # actual retrieval IDs are custom code tokens, so it can build this tokenizer
    # offline without loading google-t5/t5-small tokenizer files.
    if is_hdgr_block_denoising_config(config) or is_gpt_hdgr_config(config) or is_genius_ar_config(config):
        return False
    return True


def get_generative_retriever_class(config):
    if is_gpt_hdgr_config(config):
        from models.hdgr_comparison.retriever_gpt_hdgr import T5ForGenerativeRetrieval
        return T5ForGenerativeRetrieval
    if is_genius_ar_config(config):
        from models.hdgr_comparison.retriever_genius_ar import T5ForGenerativeRetrieval
        return T5ForGenerativeRetrieval
    raise ValueError(
        "Unsupported generator. This release retains only genius_ar and gpt_hdgr."
    )
