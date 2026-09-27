"""Streamlit UI for the TrustGraph-X v2 pipeline (local, no Colab)."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from trustgraph_pipeline import (
    DATASET_FILES,
    PipelineConfig,
    discover_datasets,
    predict_claim,
    run_pipeline,
    save_artifacts,
)

DEFAULT_DATA_DIR = os.environ.get("DATASET_DIR", str(Path(__file__).parent))

st.set_page_config(page_title="TrustGraph-X v2", page_icon="🕸️", layout="wide")


def sidebar_config() -> PipelineConfig:
    st.sidebar.header("Data")
    data_dir = st.sidebar.text_input("Dataset folder", value=DEFAULT_DATA_DIR).strip().strip('"')

    st.sidebar.header("Model")
    model_name = st.sidebar.text_input("Encoder", value="microsoft/deberta-base")
    device_options = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = st.sidebar.selectbox("Device", device_options)

    st.sidebar.header("Scale")
    train_limit = st.sidebar.number_input("Max train records", 100, 20000, 600, step=100)
    val_limit = st.sidebar.number_input("Max validation records", 50, 5000, 200, step=50)
    test_limit = st.sidebar.number_input("Max test records", 50, 5000, 200, step=50)
    max_evidence = st.sidebar.number_input("Evidence bank size", 50, 10000, 500, step=50)
    graph_nodes = st.sidebar.number_input("Semantic graph nodes", 50, 5000, 400, step=50)
    graph_k = st.sidebar.slider("Graph neighbours (K)", 2, 10, 3)
    max_len = st.sidebar.slider("Max tokens", 64, 512, 128, step=32)
    batch_size = st.sidebar.slider("Batch size", 4, 64, 16, step=4)

    st.sidebar.header("Training")
    contrastive_epochs = st.sidebar.slider("Contrastive epochs", 0, 10, 2)
    fusion_epochs = st.sidebar.slider("Fusion epochs", 1, 60, 20)
    kaggle_one_is_fake = st.sidebar.checkbox("Kaggle Target=1 means FAKE", value=True)
    run_shap = st.sidebar.checkbox("Run SHAP attribution", value=True)

    return PipelineConfig(
        data_dir=Path(data_dir),
        model_name=model_name,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        graph_k=graph_k,
        graph_nodes=int(graph_nodes),
        max_evidence=int(max_evidence),
        train_limit=int(train_limit),
        val_limit=int(val_limit),
        test_limit=int(test_limit),
        contrastive_epochs=contrastive_epochs,
        fusion_epochs=fusion_epochs,
        kaggle_one_is_fake=kaggle_one_is_fake,
        run_shap=run_shap,
    )


def render_dataset_status(cfg: PipelineConfig) -> None:
    st.subheader("Expected input datasets")
    found = discover_datasets(cfg)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "dataset": key,
                    "expected file": DATASET_FILES[key],
                    "found": str(path.relative_to(cfg.data_dir)) if path else "— missing —",
                }
                for key, path in found.items()
            ]
        ),
        use_container_width=True,
    )
    missing = [k for k, v in found.items() if v is None]
    if missing:
        st.warning(
            "Missing: " + ", ".join(missing)
            + ". Labelled datasets that are missing are skipped; if Pheme.csv is missing "
            "the semantic graph is built from the training texts instead."
        )


def render_results(result) -> None:
    metrics = result.metrics
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    cols[1].metric("Precision", f"{metrics['precision']:.4f}")
    cols[2].metric("Recall", f"{metrics['recall']:.4f}")
    cols[3].metric("F1", f"{metrics['f1']:.4f}")
    cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    tabs = st.tabs(
        ["Data", "Training", "Evaluation", "Gate analysis", "Ablation", "Explainability", "Predict"]
    )

    with tabs[0]:
        st.markdown("**Input datasets**")
        st.dataframe(result.dataset_summary, use_container_width=True)
        st.markdown("**Splits**")
        st.dataframe(result.split_summary, use_container_width=True)
        st.markdown("**Training-only evidence bank**")
        st.json(result.evidence_summary)
        st.markdown("**Semantic graph**")
        st.json(result.graph_summary)

    with tabs[1]:
        if result.contrastive_losses:
            st.markdown("**Contrastive alignment loss**")
            losses = pd.DataFrame(
                {"epoch": range(1, len(result.contrastive_losses) + 1), "loss": result.contrastive_losses}
            )
            st.plotly_chart(px.line(losses, x="epoch", y="loss", markers=True), use_container_width=True)
        else:
            st.info("Contrastive stage skipped (0 epochs or a single-class evidence bank).")
        st.markdown("**Adaptive fusion training**")
        st.plotly_chart(
            px.line(result.fusion_history, x="epoch", y=["loss", "val_accuracy", "val_f1"], markers=True),
            use_container_width=True,
        )
        st.dataframe(result.fusion_history, use_container_width=True)

    with tabs[2]:
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame(metrics["report"]).T, use_container_width=True)
        predictions = pd.DataFrame(
            {
                "label": ["Real" if v == 1 else "Fake" for v in metrics["labels"]],
                "prediction": ["Real" if v == 1 else "Fake" for v in metrics["predictions"]],
                "probability_real": metrics["probabilities"],
                "evidence_gate": metrics["gates"],
            }
        )
        confusion = (
            predictions.groupby(["label", "prediction"]).size().reset_index(name="count")
        )
        st.plotly_chart(
            px.density_heatmap(
                confusion, x="prediction", y="label", z="count", text_auto=True, color_continuous_scale="Blues"
            ),
            use_container_width=True,
        )
        st.dataframe(predictions.head(200), use_container_width=True)

    with tabs[3]:
        st.markdown(
            "Higher gate α → more weight on retrieved evidence; lower α → more weight on graph context."
        )
        st.dataframe(result.gate_analysis, use_container_width=True)
        gates = pd.DataFrame(
            {
                "label": ["Real" if v == 1 else "Fake" for v in metrics["labels"]],
                "evidence_gate": metrics["gates"],
            }
        )
        st.plotly_chart(px.histogram(gates, x="evidence_gate", color="label", nbins=30, barmode="overlay"), use_container_width=True)

    with tabs[4]:
        st.dataframe(result.ablation, use_container_width=True)
        st.plotly_chart(
            px.bar(result.ablation.melt("Model", var_name="metric", value_name="score"), x="Model", y="score", color="metric", barmode="group"),
            use_container_width=True,
        )

    with tabs[5]:
        if result.shap_summary is None:
            st.info("SHAP was disabled for this run.")
        else:
            st.markdown("**Mean |SHAP| contribution per representation block**")
            st.dataframe(result.shap_summary, use_container_width=True)
            st.plotly_chart(
                px.bar(result.shap_summary, x="Feature group", y="Mean |SHAP|"), use_container_width=True
            )

    with tabs[6]:
        claim = st.text_area("Claim / news text", height=150, key="claim_input")
        if st.button("Classify claim") and claim.strip():
            out = predict_claim(result, claim)
            left, right = st.columns(2)
            left.metric("Prediction", out["prediction"])
            left.metric("P(real)", f"{out['probability_real']:.4f}")
            right.metric("Evidence gate α", f"{out['evidence_gate']:.4f}")
            right.metric("cos(content, evidence)", f"{out['content_evidence_similarity']:.4f}")
            st.markdown("**Retrieved evidence**")
            st.dataframe(out["retrieved_evidence"], use_container_width=True)

    st.divider()
    if st.button("Save model + evidence bank"):
        paths = save_artifacts(result, Path(result.config.data_dir) / "trustgraphx_models")
        st.success("Saved: " + ", ".join(str(p) for p in paths))


def main() -> None:
    st.title("🕸️ TrustGraph-X v2 — Explainable Evidence–Graph Consistency")
    st.caption(
        "Multi-dataset input → normalization → train-only evidence bank → DeBERTa → contrastive "
        "alignment → semantic graph → graph attention → consistency gate → adaptive fusion → SHAP"
    )

    cfg = sidebar_config()

    if not cfg.data_dir.is_dir():
        st.error(f"`{cfg.data_dir}` is not an existing folder on this machine.")
        return

    render_dataset_status(cfg)

    if st.button("Run pipeline", type="primary"):
        status = st.empty()
        bar = st.progress(0.0)

        def progress(message: str, fraction: float) -> None:
            status.write(message)
            bar.progress(min(max(fraction, 0.0), 1.0))

        try:
            with st.spinner("Running TrustGraph-X..."):
                st.session_state["result"] = run_pipeline(cfg, progress)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            st.error(f"Pipeline failed: {exc}")
            st.code(traceback.format_exc())
            return
        finally:
            bar.empty()
            status.empty()

    if "result" in st.session_state:
        render_results(st.session_state["result"])
    else:
        st.info("Configure the run in the sidebar, then click **Run pipeline**.")


if __name__ == "__main__":
    main()
