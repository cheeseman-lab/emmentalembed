"""Unified PLM embedding extraction for any HuggingFace-compatible protein language model.

Works with ANY AutoModel-compatible PLM (ESM-2, ESM-C/ESM++, AMPLIFY, ProtT5,
ANKH, ProtBERT, etc.). Outputs directly to CSV and returns an EmbeddingSet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from emmentalembed.io import read_fasta
from emmentalembed.types import EmbeddingSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

_TRUST_REMOTE_CODE_PATTERNS = ("amplify", "esmplusplus", "esm++")

_ADDITIVE_MASK_PATTERNS = ("amplify",)


def _needs_trust_remote_code(model_id: str) -> bool:
    """Check whether a model ID likely requires trust_remote_code=True."""
    return any(p in model_id.lower() for p in _TRUST_REMOTE_CODE_PATTERNS)


def _needs_additive_mask(model_id: str) -> bool:
    """Check whether a model expects additive attention masks (0, -inf)."""
    return any(p in model_id.lower() for p in _ADDITIVE_MASK_PATTERNS)


def _is_encoder_decoder(model) -> bool:
    """Check whether a model is an encoder-decoder (e.g. T5-based Ankh, ProtT5)."""
    return getattr(model.config, "is_encoder_decoder", False)


def _needs_spaced_sequences(tokenizer) -> bool:
    """Detect whether a tokenizer requires spaces between amino acid characters.

    Many protein LMs (ProtBERT, ProtT5) use character-level vocabularies that
    expect single-character tokens separated by spaces.
    """
    probe = "ACDEFG"
    try:
        n_raw = len(tokenizer.encode(probe))
        n_spaced = len(tokenizer.encode(" ".join(list(probe))))
    except Exception:
        return False
    return n_raw < 4 and n_spaced >= 6


def _fix_meta_tensors(model) -> None:
    """Recompute model attributes stuck on the meta device.

    Newer transformers versions initialise models on the meta device then
    materialise only checkpoint-stored parameters. Computed attributes like
    rotary embedding frequency tables remain as meta tensors.
    """
    import sys

    import torch

    if hasattr(model, "freqs_cis") and model.freqs_cis.device.type == "meta":
        precompute_fn = None
        for mod_name, mod in sys.modules.items():
            if "rotary" in mod_name.lower() and hasattr(mod, "precompute_freqs_cis"):
                precompute_fn = mod.precompute_freqs_cis
                break

        if precompute_fn is not None:
            d_head = model.config.hidden_size // model.config.num_attention_heads
            max_len = getattr(model.config, "max_length", 2048)
            model.freqs_cis = precompute_fn(d_head, max_len)
            logger.info("Recomputed freqs_cis (was on meta device)")
        else:
            logger.warning(
                "Model has freqs_cis on meta device but precompute_freqs_cis "
                "function not found; forward pass may fail"
            )

    for attr_name in list(vars(model)):
        attr = getattr(model, attr_name, None)
        if isinstance(attr, torch.Tensor) and attr.device.type == "meta":
            if attr_name != "freqs_cis":
                logger.warning("Model attribute %r is on meta device and was not fixed", attr_name)


def _load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    device: str,
    device_map: str | None = None,
):
    """Load a HuggingFace model and tokenizer suitable for embedding extraction."""
    import torch
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, torch.float16)

    trust_remote_code = _needs_trust_remote_code(model_id)

    # T5-based models (ProtT5, ANKH) use SentencePiece tokenizers that fail
    # with the fast (tiktoken) converter — force the slow tokenizer for them.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        logger.info("Fast tokenizer failed for %s, falling back to slow tokenizer", model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

    model_classes = [AutoModel, AutoModelForMaskedLM, AutoModelForSeq2SeqLM]
    last_error = None

    for model_cls in model_classes:
        try:
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            model = model_cls.from_pretrained(model_id, **load_kwargs)
            break
        except (ValueError, OSError, KeyError, AttributeError) as exc:
            last_error = exc
            logger.debug("Failed to load %s with %s: %s", model_id, model_cls.__name__, exc)
            continue
    else:
        raise RuntimeError(
            f"Could not load model {model_id!r} with AutoModel, "
            f"AutoModelForMaskedLM, or AutoModelForSeq2SeqLM. Last error: {last_error}"
        )

    _fix_meta_tensors(model)

    if device_map is None:
        model = model.to(device)
    model.eval()

    logger.info(
        "Loaded %s (%s) on %s (dtype=%s, trust_remote_code=%s)",
        model_id,
        type(model).__name__,
        device,
        torch_dtype,
        trust_remote_code,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Encoder-decoder helpers
# ---------------------------------------------------------------------------


def _encoder_forward(model, input_ids, attention_mask, use_amp, torch_dtype_for_amp):
    """Extract encoder hidden states from a T5-style encoder-decoder model."""
    import torch

    encoder = model.get_encoder()
    if use_amp:
        with torch.cuda.amp.autocast(dtype=torch_dtype_for_amp):
            enc_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    else:
        enc_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return enc_outputs.hidden_states


# ---------------------------------------------------------------------------
# ESM-C (EvolutionaryScale esm SDK) extraction
# ---------------------------------------------------------------------------

_ESMC_PATTERNS = ("esmc-", "esmc_")


def _is_esmc_model(model_id: str) -> bool:
    """Check whether a model ID refers to an ESM-C model from EvolutionaryScale."""
    return any(p in model_id.lower() for p in _ESMC_PATTERNS)


def _extract_esmc(
    fasta_path: str | Path,
    model_id: str,
    output_path: str | Path | None,
    layer: int,
    representation: str,
    batch_size: int,
    device: str,
) -> EmbeddingSet:
    """Extract embeddings using the EvolutionaryScale ``esm`` SDK."""
    import torch
    from tqdm import tqdm

    try:
        from esm.models.esmc import ESMC
    except ImportError as exc:
        raise ImportError(
            "ESM-C models require the 'esm' package. Install with: pip install esm"
        ) from exc

    sequences = read_fasta(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")

    variant_ids = list(sequences.keys())
    seqs = list(sequences.values())

    raw_name = model_id.split("/")[-1].lower().replace("-", "_")
    if "300m" in raw_name:
        sdk_name = "esmc_300m"
    elif "600m" in raw_name:
        sdk_name = "esmc_600m"
    else:
        sdk_name = raw_name
    model = ESMC.from_pretrained(sdk_name).to(device).eval()
    tokenizer = model.tokenizer

    all_embeddings: list[np.ndarray] = []
    num_batches = (len(seqs) + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, len(seqs), batch_size),
        total=num_batches,
        desc=f"Extracting embeddings ({model_id})",
    ):
        batch_seqs = seqs[batch_start : batch_start + batch_size]

        encoded = tokenizer(batch_seqs, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        if layer == -1:
            layer_output = outputs.embeddings
        else:
            layer_output = outputs.hidden_states[layer]

        special_ids = set(tokenizer.all_special_ids)
        for i in range(layer_output.size(0)):
            token_embeddings = layer_output[i]
            mask = attention_mask[i]

            if representation == "mean":
                token_ids = input_ids[i]
                content_mask = torch.tensor(
                    [
                        (mask[j].item() == 1 and token_ids[j].item() not in special_ids)
                        for j in range(mask.size(0))
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                if content_mask.any():
                    pooled = token_embeddings[content_mask].mean(dim=0)
                else:
                    pooled = token_embeddings[mask.bool()].mean(dim=0)
                all_embeddings.append(pooled.float().cpu().numpy())
            elif representation == "bos":
                all_embeddings.append(token_embeddings[0].float().cpu().numpy())
            elif representation == "per_token":
                per_tok = token_embeddings[mask.bool()].float().cpu().numpy().flatten()
                all_embeddings.append(per_tok)

        del outputs, layer_output, input_ids, attention_mask, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embeddings_df = pd.DataFrame(all_embeddings, index=variant_ids)
    embeddings_df.index.name = "variant_id"

    logger.info(
        "Extracted ESM-C embeddings: shape=%s, model=%s, layer=%d, representation=%s",
        embeddings_df.shape,
        model_id,
        layer,
        representation,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(output_path)
        logger.info("Saved embeddings to %s", output_path)

    return EmbeddingSet(
        embeddings=embeddings_df,
        model_name=model_id,
        layer=layer,
        representation=representation,
    )


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    fasta_path: str | Path,
    model_id: str = "facebook/esm2_t33_650M_UR50D",
    output_path: str | Path | None = None,
    layer: int = -1,
    representation: str = "mean",
    batch_size: int = 32,
    dtype: str = "float16",
    device: str = "cuda",
    device_map: str | None = None,
) -> EmbeddingSet:
    """Extract PLM embeddings from a FASTA file using any HuggingFace protein model.

    Works with ESM-2 (any size), ESM-C (via the ``esm`` SDK), AMPLIFY,
    ProtBERT, ProtT5, ANKH, and any other model loadable via
    ``AutoModel.from_pretrained``.

    Args:
        fasta_path: Path to FASTA file containing protein sequences.
        model_id: HuggingFace model identifier or local path.
        output_path: If provided, save the embedding CSV here.
        layer: Hidden-state layer index to extract. ``-1`` means the last layer.
        representation: Pooling strategy (mean, bos, per_token).
        batch_size: Number of sequences per forward pass.
        dtype: Model precision (float16, bfloat16, float32).
        device: PyTorch device string.
        device_map: HuggingFace device map for model sharding.

    Returns:
        An EmbeddingSet containing the extracted embeddings and metadata.
    """
    import torch
    from tqdm import tqdm

    if representation not in ("mean", "bos", "per_token"):
        raise ValueError(
            f"representation must be 'mean', 'bos', or 'per_token', got {representation!r}"
        )

    # ESM-C models use a separate SDK
    if _is_esmc_model(model_id):
        return _extract_esmc(
            fasta_path, model_id, output_path, layer, representation, batch_size, device
        )

    # Read sequences
    sequences = read_fasta(fasta_path)
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")

    variant_ids = list(sequences.keys())
    seqs = list(sequences.values())

    # Load model
    model, tokenizer = _load_model_and_tokenizer(model_id, dtype, device, device_map=device_map)

    # Detect model-specific quirks
    additive_mask = _needs_additive_mask(model_id)
    encoder_decoder = _is_encoder_decoder(model)
    spaced = _needs_spaced_sequences(tokenizer)

    if encoder_decoder:
        logger.info("Detected encoder-decoder model; extracting encoder hidden states only")
    if spaced:
        logger.info("Tokenizer requires space-separated amino acid characters")

    # Handle <eos> chain separator
    if any("<eos>" in s for s in seqs):
        eos_is_special = "<eos>" in tokenizer.get_vocab()
        if eos_is_special:
            logger.info("Tokenizer recognises <eos> — keeping chain separator")
        else:
            seqs = [s.replace("<eos>", "") for s in seqs]
            logger.info("Tokenizer does not recognise <eos> — stripping chain separator")

    # Insert spaces if needed
    if spaced:
        seqs = [" ".join(list(s)) for s in seqs]

    # Batched extraction
    all_embeddings: list[np.ndarray] = []

    use_amp = dtype == "float16" and device != "cpu"
    torch_dtype_for_amp = torch.float16

    num_batches = (len(seqs) + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, len(seqs), batch_size),
        total=num_batches,
        desc=f"Extracting embeddings ({model_id})",
    ):
        batch_seqs = seqs[batch_start : batch_start + batch_size]

        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        fwd_mask = attention_mask
        if additive_mask:
            fwd_mask = torch.where(
                attention_mask.bool(),
                torch.tensor(0.0, device=device),
                torch.tensor(float("-inf"), device=device),
            )

        with torch.no_grad():
            if encoder_decoder:
                hidden_states = _encoder_forward(
                    model, input_ids, attention_mask, use_amp, torch_dtype_for_amp
                )
            else:
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch_dtype_for_amp):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=fwd_mask,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=fwd_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden_states = outputs.hidden_states

        layer_output = hidden_states[layer]

        for i in range(layer_output.size(0)):
            token_embeddings = layer_output[i]
            mask = attention_mask[i]

            if representation == "mean":
                special_ids = set(tokenizer.all_special_ids)
                token_ids = input_ids[i]
                content_mask = torch.tensor(
                    [
                        (mask[j].item() == 1 and token_ids[j].item() not in special_ids)
                        for j in range(mask.size(0))
                    ],
                    dtype=torch.bool,
                    device=device,
                )

                if content_mask.any():
                    pooled = token_embeddings[content_mask].mean(dim=0)
                else:
                    attended = mask.bool()
                    pooled = token_embeddings[attended].mean(dim=0)

                all_embeddings.append(pooled.float().cpu().numpy())

            elif representation == "bos":
                bos_emb = token_embeddings[0]
                all_embeddings.append(bos_emb.float().cpu().numpy())

            elif representation == "per_token":
                attended = mask.bool()
                per_tok = token_embeddings[attended].float().cpu().numpy().flatten()
                all_embeddings.append(per_tok)

        cleanup_vars = [hidden_states, layer_output, input_ids, attention_mask, encoded]
        if not encoder_decoder:
            cleanup_vars.append(outputs)
        del cleanup_vars
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Assemble DataFrame
    embeddings_df = pd.DataFrame(all_embeddings, index=variant_ids)
    embeddings_df.index.name = "variant_id"

    logger.info(
        "Extracted embeddings: shape=%s, model=%s, layer=%d, representation=%s",
        embeddings_df.shape,
        model_id,
        layer,
        representation,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.to_parquet(output_path)
        logger.info("Saved embeddings to %s", output_path)

    return EmbeddingSet(
        embeddings=embeddings_df,
        model_name=model_id,
        layer=layer,
        representation=representation,
    )
