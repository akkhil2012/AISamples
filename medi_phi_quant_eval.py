"""
Sequential quantization evaluation for microsoft/MediPhi.

Applies several torchao quantization configs to MediPhi one at a time
(lightest to most aggressive) and regenerates a fixed prompt's completion
after each one, so output quality can be compared side by side before
committing to a config for the real ExecuTorch export
(see medi_phi_executorch_mod.py).

This script does NOT export to ExecuTorch -- it's purely for picking a
quantization config by eyeballing generation quality.

Memory note: MediPhi is ~3.8B params. This machine has 16GB of RAM, so we
load it in bfloat16 (not float32) and reload it FRESH from the HF cache for
every config rather than keeping a pristine copy around to deepcopy --
quantize_() mutates the model in place and can't be undone, and a second
live copy of the model would roughly double peak memory. Reloading from the
local HF cache is cheap (the weights are already on disk / in the OS page
cache); only one full model instance is ever resident at a time.
"""

import gc
import math

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

MODEL_NAME = "microsoft/MediPhi"
PROMPT = "What are the symptoms of type 2 diabetes?"
MAX_NEW_TOKENS = 64

# Fixed (prompt, reference completion) pairs used to score each config by
# teacher-forced perplexity, instead of judging quality from a single
# greedy sample. References are short, factually-checked continuations a
# medical model should assign high likelihood to.
REFERENCE_SET = [
    (
        "What are the symptoms of type 2 diabetes?",
        " Common symptoms include increased thirst, frequent urination, increased "
        "hunger, fatigue, blurred vision, slow-healing sores, and unexplained weight loss.",
    ),
    (
        "What is the recommended first-line medication for type 2 diabetes?",
        " Metformin is typically the recommended first-line medication, as it helps "
        "lower blood sugar levels and has a favorable safety profile.",
    ),
    (
        "How is type 1 diabetes different from type 2 diabetes?",
        " Type 1 diabetes is an autoimmune condition where the pancreas produces "
        "little or no insulin, while type 2 diabetes involves insulin resistance and "
        "is often linked to lifestyle factors.",
    ),
    (
        "What lifestyle changes can help manage type 2 diabetes?",
        " Regular exercise, a balanced diet low in refined sugars, weight management, "
        "and monitoring blood glucose levels can all help manage the condition.",
    ),
]

# Tried in order, lightest to most aggressive. `None` means "skip
# quantize_() entirely" -- the unquantized reference output.
QUANT_CONFIGS = [
    ("baseline (bf16, no quantization)", None),
    ("int8 weight-only", Int8WeightOnlyConfig()),
    ("int4 weight-only (group=32)", Int4WeightOnlyConfig(group_size=32)),
    ("int8 dynamic activation + int8 weight", Int8DynamicActivationInt8WeightConfig()),
    (
        "int8 dynamic activation + int4 weight (group=32)",  # used by medi_phi_executorch_mod.py
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        ),
    ),
]


def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_completion(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy, so runs are comparable across configs
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compute_perplexity(model, tokenizer, reference_set):
    """Teacher-forced perplexity of `model` over `reference_set`.

    Only the reference completion tokens are scored -- prompt tokens are
    masked out of the loss -- so this measures how well the (possibly
    quantized) model predicts a fixed, human-written answer, independent
    of whatever it would greedily generate on its own.
    """
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for prompt, reference in reference_set:
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            inputs = tokenizer(prompt + reference, return_tensors="pt")
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_len] = -100
            outputs = model(**inputs, labels=labels)
            n_scored = (labels != -100).sum().item()
            total_nll += outputs.loss.item() * n_scored
            total_tokens += n_scored
    return math.exp(total_nll / total_tokens)


def main():
    results = []

    progress = tqdm(QUANT_CONFIGS, desc="Quantization configs", unit="config")
    for label, config in progress:
        progress.set_postfix_str(label)
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        print("Loading fresh model copy...")
        model, tokenizer = load_model_and_tokenizer()

        if config is not None:
            print(f"Applying quantize_() with {config!r}")
            try:
                quantize_(model, config)
            except (ImportError, NotImplementedError) as e:
                print(f"Skipping {label!r}: {e}")
                del model, tokenizer
                gc.collect()
                continue
        else:
            print("No quantization applied (reference output).")

        print("Generating completion...")
        completion = generate_completion(model, tokenizer, PROMPT)
        print(f"Prompt: {PROMPT}")
        print(f"Completion: {completion}")

        print("Scoring reference set...")
        perplexity = compute_perplexity(model, tokenizer, REFERENCE_SET)
        print(f"Perplexity on reference set ({len(REFERENCE_SET)} examples): {perplexity:.3f}")

        results.append((label, completion, perplexity))

        # Drop the model before loading the next one -- see memory note above.
        del model, tokenizer
        gc.collect()

    print(f"\n{'=' * 70}\nSummary for prompt: {PROMPT!r}\n{'=' * 70}")
    for label, completion, perplexity in results:
        print(f"\n[{label}] perplexity={perplexity:.3f}\n{completion}")

    print(f"\n{'=' * 70}\nRanked by perplexity (lower = better fit to reference set)\n{'=' * 70}")
    for label, completion, perplexity in sorted(results, key=lambda r: r[2]):
        print(f"{perplexity:8.3f}  {label}")


if __name__ == "__main__":
    main()
