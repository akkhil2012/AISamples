"""
Streamlit UI for medi_phi_quant_eval.py.

Run with:
    streamlit run quantization/medi_phi_quant_eval_app.py

Wraps the same sequential quantization sweep as the CLI script in an
interactive UI: pick which configs to run, watch progress live, and see
each config's greedy completion plus reference-set perplexity as it
finishes. Reuses the CLI script's functions/configs directly so the two
never drift apart.
"""

import gc
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import medi_phi_quant_eval as mpqe
from medi_phi_quant_eval import (
    MAX_NEW_TOKENS,
    PROMPT,
    QUANT_CONFIGS,
    REFERENCE_SET,
    compute_perplexity,
    generate_completion,
    load_model_and_tokenizer,
    quantize_,
)

st.set_page_config(page_title="MediPhi Quantization Eval", layout="wide")
st.title("MediPhi Quantization Eval")
st.caption(
    "Sequentially applies torchao quantization configs to microsoft/MediPhi and "
    "compares greedy completions plus reference-set perplexity, so you can pick a "
    "config before exporting to ExecuTorch (see medi_phi_executorch_mod.py)."
)

prompt = st.text_input("Prompt", value=PROMPT)
max_new_tokens = st.number_input(
    "Max new tokens", min_value=1, max_value=512, value=MAX_NEW_TOKENS
)

labels = [label for label, _ in QUANT_CONFIGS]
selected_labels = st.multiselect("Configs to run", labels, default=labels)

run = st.button("Run evaluation", type="primary")

if "results" not in st.session_state:
    st.session_state.results = []

if run:
    selected = [(label, config) for label, config in QUANT_CONFIGS if label in selected_labels]
    if not selected:
        st.warning("Select at least one config to run.")
    else:
        mpqe.MAX_NEW_TOKENS = max_new_tokens
        st.session_state.results = []
        progress_bar = st.progress(0.0)
        status = st.status("Starting...", expanded=True)
        results_container = st.container()

        for i, (label, config) in enumerate(selected):
            status.update(label=f"[{i + 1}/{len(selected)}] {label}")
            status.write("Loading fresh model copy...")
            model, tokenizer = load_model_and_tokenizer()

            skip_reason = None
            if config is not None:
                status.write(f"Applying `quantize_()` with `{config!r}`")
                try:
                    quantize_(model, config)
                except (ImportError, NotImplementedError) as e:
                    skip_reason = str(e)
            else:
                status.write("No quantization applied (reference output).")

            if skip_reason is not None:
                status.write(f":warning: Skipping **{label}**: {skip_reason}")
                del model, tokenizer
                gc.collect()
                progress_bar.progress((i + 1) / len(selected))
                continue

            status.write("Generating completion...")
            completion = generate_completion(model, tokenizer, prompt)

            status.write("Scoring reference set...")
            perplexity = compute_perplexity(model, tokenizer, REFERENCE_SET)

            del model, tokenizer
            gc.collect()

            st.session_state.results.append(
                {"label": label, "perplexity": perplexity, "completion": completion}
            )
            with results_container:
                st.subheader(f"{label} — perplexity {perplexity:.3f}")
                st.write(completion)

            progress_bar.progress((i + 1) / len(selected))

        status.update(label="Done", state="complete")

if st.session_state.results:
    st.header("Summary (ranked by perplexity, lower = better fit to reference set)")
    df = pd.DataFrame(st.session_state.results).sort_values("perplexity")
    st.dataframe(df, use_container_width=True, hide_index=True)
