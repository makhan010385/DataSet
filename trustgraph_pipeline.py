"""TrustGraph-X v2 pipeline: local (non-Colab) implementation.

Evidence-Graph Consistency Adaptive Fusion for fake news detection:
multi-dataset input -> normalization -> train-only evidence bank -> DeBERTa ->
contrastive alignment -> semantic graph -> graph attention -> consistency gate ->
adaptive fusion -> fake/real -> SHAP.

Every stage is a plain function so the Streamlit app (or a script) can drive it.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DATASET_FILES = {
    "gossipcop": "gossipcop.csv",
    "isot": "ISOT.csv",
    "kaggle": "kaggle.csv",
    "politifact": "PolitiFact.csv",
    "snopes": "snopeswithsum.csv",
    "welfake": "WELFake_Dataset.csv",
    "pheme": "Pheme.csv",
}
LABELLED_SOURCES = ["gossipcop", "isot", "kaggle", "welfake", "politifact", "snopes"]
EVIDENCE_SOURCES = ["politifact", "snopes"]

TEXT_COLUMNS = ["text", "statement", "content", "article", "title", "tweet", "claim"]
LABEL_COLUMNS = ["label", "target", "class", "binarytarget", "binarynumtarget", "y"]

Progress = Callable[[str, float], None]


def _noop(message: str, fraction: float) -> None:
    del message, fraction


@dataclass
class PipelineConfig:
    data_dir: Path
    model_name: str = "microsoft/deberta-base"
    max_len: int = 128
    batch_size: int = 16
    graph_k: int = 3
    graph_nodes: int = 1000
    max_evidence: int = 3000
    train_limit: int = 3000
    val_limit: int = 1000
    test_limit: int = 1000
    evidence_top_k: int = 3
    contrastive_epochs: int = 3
    fusion_epochs: int = 30
    kaggle_one_is_fake: bool = True
    seed: int = 42
    device: str = "cpu"
    run_shap: bool = True
    shap_samples: int = 5
    files: dict[str, str] = field(default_factory=lambda: dict(DATASET_FILES))

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Stage 1: dataset discovery and loading
# --------------------------------------------------------------------------- #
def find_dataset_file(data_dir: Path, filename: str) -> Path | None:
    """Locate `filename` in `data_dir` (case-insensitive, recursive)."""
    direct = data_dir / filename
    if direct.is_file():
        return direct
    target = filename.lower()
    stem = Path(filename).stem.lower()
    fallback: Path | None = None
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == target:
            return path
        if fallback is None and path.suffix.lower() == ".csv" and stem in name:
            fallback = path
    return fallback


def discover_datasets(cfg: PipelineConfig) -> dict[str, Path | None]:
    return {key: find_dataset_file(cfg.data_dir, name) for key, name in cfg.files.items()}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", on_bad_lines="skip", low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    return df.dropna(axis=1, how="all")


# --------------------------------------------------------------------------- #
# Stage 2: normalization
# --------------------------------------------------------------------------- #
def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip()


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalize_binary_label(value: object, dataset: str, kaggle_one_is_fake: bool) -> float:
    """Map a dataset-specific label onto 0 = fake, 1 = real."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()

    if text in {"fake", "false", "mostly false", "pants on fire", "fake news"}:
        return 0.0
    if text in {"real", "true", "mostly true", "correct attribution", "truth"}:
        return 1.0

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    if number not in (0, 1):
        return np.nan
    if dataset == "kaggle" and kaggle_one_is_fake:
        return float(1 - number)
    return float(number)


def standardize(df: pd.DataFrame, dataset: str, cfg: PipelineConfig) -> pd.DataFrame:
    text_col = find_col(df, TEXT_COLUMNS)
    label_col = find_col(df, LABEL_COLUMNS)
    if text_col is None:
        raise ValueError(f"{dataset}: no text column found in {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"{dataset}: no label column found in {list(df.columns)}")

    out = pd.DataFrame(
        {
            "text": df[text_col].map(clean_text),
            "label": df[label_col].map(
                lambda v: normalize_binary_label(v, dataset, cfg.kaggle_one_is_fake)
            ),
            "source": dataset,
        }
    )
    out = out[out["text"].str.len() >= 20].dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out["text_norm"] = (
        out["text"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out.drop_duplicates("text_norm").drop(columns="text_norm").reset_index(drop=True)


def build_master(cfg: PipelineConfig, standardized: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, frame in standardized.items():
        data = frame
        if len(data) > cfg.train_limit and data["label"].nunique() > 1:
            data, _ = train_test_split(
                data,
                train_size=cfg.train_limit,
                stratify=data["label"],
                random_state=cfg.seed,
            )
        elif len(data) > cfg.train_limit:
            data = data.sample(cfg.train_limit, random_state=cfg.seed)
        parts.append(data.assign(source=name))

    master = pd.concat(parts, ignore_index=True)
    master["text_norm"] = (
        master["text"]
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return master.drop_duplicates("text_norm").drop(columns="text_norm").reset_index(drop=True)


def split_master(
    cfg: PipelineConfig, master: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        master, test_size=0.20, stratify=master["label"], random_state=cfg.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=cfg.seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Stage 3: DeBERTa encoder with an embedding cache
# --------------------------------------------------------------------------- #
class TextEncoder:
    def __init__(self, cfg: PipelineConfig):
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.device = cfg.torch_device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(self.device).eval()
        self.hidden_size = self.model.config.hidden_size
        self.cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: list[str], progress: Progress = _noop) -> torch.Tensor:
        embeddings = []
        total = max(1, len(texts))
        for start in range(0, len(texts), self.cfg.batch_size):
            batch = texts[start : start + self.cfg.batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_len,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            pooled = self.mean_pool(hidden, tokens["attention_mask"])
            embeddings.append(F.normalize(pooled, p=2, dim=1).cpu())
            progress("Encoding with DeBERTa", min(1.0, (start + len(batch)) / total))
        return torch.cat(embeddings, dim=0)

    def embed(self, texts: Iterable[str], progress: Progress = _noop) -> torch.Tensor:
        texts = list(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for text, vector in zip(missing, self.encode(missing, progress)):
                self.cache[text] = vector
        return torch.stack([self.cache[t] for t in texts])


# --------------------------------------------------------------------------- #
# Stage 4-5: evidence bank, retrieval and semantic graph
# --------------------------------------------------------------------------- #
class EvidenceBank:
    """Training-only evidence bank; validation/test texts are never inserted."""

    def __init__(self, frame: pd.DataFrame, vectors: torch.Tensor):
        self.frame = frame.reset_index(drop=True)
        self.vectors = F.normalize(vectors, p=2, dim=1)

    @classmethod
    def build(
        cls, cfg: PipelineConfig, train_df: pd.DataFrame, encoder: TextEncoder, progress: Progress
    ) -> "EvidenceBank":
        frame = train_df[train_df["source"].isin(EVIDENCE_SOURCES)]
        if frame.empty:
            frame = train_df
        frame = frame[["text", "label", "source"]]
        if len(frame) > cfg.max_evidence:
            frame = frame.sample(cfg.max_evidence, random_state=cfg.seed)
        frame = frame.reset_index(drop=True)
        vectors = encoder.embed(frame["text"].tolist(), progress)
        return cls(frame, vectors)

    @torch.no_grad()
    def retrieve(
        self, encoder: TextEncoder, texts: list[str], top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        query = F.normalize(encoder.embed(texts), p=2, dim=1)
        scores = query @ self.vectors.T
        values, indices = torch.topk(scores, k=min(top_k, self.vectors.shape[0]), dim=1)
        return indices.numpy(), values.numpy()


class SemanticGraph:
    """K-NN graph over DeBERTa embeddings (PHEME when available)."""

    def __init__(self, cfg: PipelineConfig, texts: list[str], vectors: np.ndarray):
        self.cfg = cfg
        self.texts = texts
        self.vectors = vectors
        self.index = NearestNeighbors(
            n_neighbors=min(cfg.graph_k + 1, len(vectors)), metric="cosine"
        ).fit(vectors)
        distances, _ = self.index.kneighbors(vectors)
        self.mean_neighbor_similarity = float(1 - distances[:, 1:].mean()) if len(vectors) > 1 else 1.0
        self.tensor = torch.tensor(vectors, dtype=torch.float32)

    @classmethod
    def build(
        cls,
        cfg: PipelineConfig,
        graph_df: pd.DataFrame,
        encoder: TextEncoder,
        progress: Progress,
    ) -> "SemanticGraph":
        frame = graph_df.drop_duplicates("text").reset_index(drop=True)
        if len(frame) > cfg.graph_nodes:
            frame = frame.sample(cfg.graph_nodes, random_state=cfg.seed).reset_index(drop=True)
        texts = frame["text"].tolist()
        vectors = encoder.embed(texts, progress).numpy()
        return cls(cfg, texts, vectors)

    def neighbors(self, encoder: TextEncoder, texts: list[str], device: torch.device) -> torch.Tensor:
        query = encoder.embed(texts).numpy()
        k = min(self.cfg.graph_k, len(self.vectors))
        indices = self.index.kneighbors(query, n_neighbors=k, return_distance=False)
        return self.tensor[indices].to(device)


# --------------------------------------------------------------------------- #
# Stage 6: contrastive alignment
# --------------------------------------------------------------------------- #
class ContrastiveProjector(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


def info_nce_loss(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)
    pos = (anchor * positive).sum(1, keepdim=True) / temperature
    neg = (anchor * negative).sum(1, keepdim=True) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def train_contrastive(
    cfg: PipelineConfig,
    encoder: TextEncoder,
    bank: EvidenceBank,
    train_small: pd.DataFrame,
    progress: Progress,
) -> tuple[ContrastiveProjector, list[float]]:
    device = cfg.torch_device
    projector = ContrastiveProjector(encoder.hidden_size).to(device)

    pos_pool = bank.frame[bank.frame.label == 1]
    neg_pool = bank.frame[bank.frame.label == 0]
    if pos_pool.empty or neg_pool.empty:
        return projector, []

    positives, negatives = [], []
    for label in train_small["label"]:
        same, other = (pos_pool, neg_pool) if label == 1 else (neg_pool, pos_pool)
        positives.append(same.sample(1).iloc[0]["text"])
        negatives.append(other.sample(1).iloc[0]["text"])

    optimizer = torch.optim.AdamW(projector.parameters(), lr=2e-4, weight_decay=1e-4)
    losses = []
    batch = 32
    for epoch in range(1, cfg.contrastive_epochs + 1):
        order = np.random.permutation(len(train_small))
        total, steps = 0.0, 0
        projector.train()
        for start in range(0, len(order), batch):
            ids = order[start : start + batch]
            anchor = projector(encoder.embed(train_small.iloc[ids]["text"].tolist()).to(device))
            pos = projector(encoder.embed([positives[i] for i in ids]).to(device))
            neg = projector(encoder.embed([negatives[i] for i in ids]).to(device))
            loss = info_nce_loss(anchor, pos, neg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        losses.append(total / max(1, steps))
        progress(f"Contrastive epoch {epoch}/{cfg.contrastive_epochs}", epoch / cfg.contrastive_epochs)
    return projector, losses


# --------------------------------------------------------------------------- #
# Stage 7-8: graph attention, consistency gate, adaptive fusion
# --------------------------------------------------------------------------- #
class TrustGraphX(nn.Module):
    def __init__(self, hidden_size: int, graph_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.content_proj = nn.Linear(hidden_size, graph_hidden)
        self.evidence_proj = nn.Linear(hidden_size, graph_hidden)
        self.graph_proj = nn.Linear(hidden_size, graph_hidden)
        self.q = nn.Linear(graph_hidden, graph_hidden, bias=False)
        self.k = nn.Linear(graph_hidden, graph_hidden, bias=False)
        self.v = nn.Linear(graph_hidden, graph_hidden, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(graph_hidden * 2 + 2, graph_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, 1),
            nn.Sigmoid(),
        )

    def graph_attention(self, content: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        q = self.q(content).unsqueeze(1)
        k = self.k(neighbors)
        v = self.v(neighbors)
        attn = torch.softmax((q * k).sum(-1) / math.sqrt(content.size(-1)), dim=1)
        return (attn.unsqueeze(-1) * v).sum(1)

    def forward(
        self, content: torch.Tensor, evidence: torch.Tensor, neighbors: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        c = F.relu(self.content_proj(content))
        e = F.relu(self.evidence_proj(evidence))
        n = F.relu(self.graph_proj(neighbors))
        g = self.graph_attention(c, n)

        ce = F.cosine_similarity(c, e, dim=1).unsqueeze(1)
        eg = F.cosine_similarity(e, g, dim=1).unsqueeze(1)
        alpha = self.gate(torch.cat([e, g, ce, eg], dim=1))

        return {
            "content": c,
            "evidence": e,
            "graph": g,
            "gate": alpha,
            "ce_similarity": ce,
            "eg_similarity": eg,
        }


class AdaptiveFusionClassifier(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3 + 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

    def forward(
        self,
        content: torch.Tensor,
        evidence: torch.Tensor,
        graph: torch.Tensor,
        sim: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([content, evidence, graph, sim], dim=1))


@dataclass
class Features:
    content: torch.Tensor
    evidence: torch.Tensor
    graph: torch.Tensor
    sim: torch.Tensor
    gate: torch.Tensor
    labels: torch.Tensor


@torch.no_grad()
def make_features(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    encoder: TextEncoder,
    bank: EvidenceBank,
    graph: SemanticGraph,
    model: TrustGraphX,
    progress: Progress,
    stage: str = "Building features",
) -> Features:
    model.eval()
    device = cfg.torch_device
    chunks: dict[str, list[torch.Tensor]] = {k: [] for k in ("c", "e", "g", "sim", "gate")}
    labels: list[int] = []
    total = max(1, len(df))

    for start in range(0, len(df), cfg.batch_size):
        batch = df.iloc[start : start + cfg.batch_size]
        contents = batch["text"].tolist()
        indices, _ = bank.retrieve(encoder, contents, top_k=1)
        evidence_texts = [bank.frame.iloc[int(indices[i, 0])]["text"] for i in range(len(contents))]

        out = model(
            encoder.embed(contents).to(device),
            encoder.embed(evidence_texts).to(device),
            graph.neighbors(encoder, contents, device),
        )
        chunks["c"].append(out["content"].cpu())
        chunks["e"].append(out["evidence"].cpu())
        chunks["g"].append(out["graph"].cpu())
        chunks["sim"].append(torch.cat([out["ce_similarity"], out["eg_similarity"]], dim=1).cpu())
        chunks["gate"].append(out["gate"].cpu())
        labels.extend(batch["label"].astype(int).tolist())
        progress(stage, min(1.0, (start + len(batch)) / total))

    return Features(
        content=torch.cat(chunks["c"]),
        evidence=torch.cat(chunks["e"]),
        graph=torch.cat(chunks["g"]),
        sim=torch.cat(chunks["sim"]),
        gate=torch.cat(chunks["gate"]),
        labels=torch.tensor(labels),
    )


def train_fusion(
    cfg: PipelineConfig, train: Features, val: Features, progress: Progress
) -> tuple[AdaptiveFusionClassifier, pd.DataFrame]:
    device = cfg.torch_device
    fusion = AdaptiveFusionClassifier(train.content.shape[1]).to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    tr = {k: getattr(train, k).to(device) for k in ("content", "evidence", "graph", "sim", "gate")}
    va = {k: getattr(val, k).to(device) for k in ("content", "evidence", "graph", "sim", "gate")}
    y_tr = train.labels.to(device)
    y_va = val.labels.numpy()

    history, best_f1, best_state = [], -1.0, None
    for epoch in range(1, cfg.fusion_epochs + 1):
        fusion.train()
        order = torch.randperm(len(y_tr), device=device)
        total, steps = 0.0, 0
        for start in range(0, len(order), 32):
            ids = order[start : start + 32]
            logits = fusion(
                tr["content"][ids],
                tr["evidence"][ids] * tr["gate"][ids],
                tr["graph"][ids] * (1.0 - tr["gate"][ids]),
                tr["sim"][ids],
            )
            loss = criterion(logits, y_tr[ids])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1

        fusion.eval()
        with torch.no_grad():
            predictions = (
                fusion(
                    va["content"],
                    va["evidence"] * va["gate"],
                    va["graph"] * (1.0 - va["gate"]),
                    va["sim"],
                )
                .argmax(1)
                .cpu()
                .numpy()
            )
        _, _, f1, _ = precision_recall_fscore_support(
            y_va, predictions, average="binary", zero_division=0
        )
        accuracy = accuracy_score(y_va, predictions)
        history.append(
            {"epoch": epoch, "loss": total / max(1, steps), "val_accuracy": accuracy, "val_f1": f1}
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}
        progress(f"Fusion epoch {epoch}/{cfg.fusion_epochs}", epoch / cfg.fusion_epochs)

    if best_state is not None:
        fusion.load_state_dict(best_state)
    return fusion, pd.DataFrame(history)


# --------------------------------------------------------------------------- #
# Stage 9-10: evaluation, ablation, SHAP
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    cfg: PipelineConfig, features: Features, fusion: AdaptiveFusionClassifier
) -> dict[str, object]:
    device = cfg.torch_device
    gate = features.gate.to(device)
    logits = fusion(
        features.content.to(device),
        features.evidence.to(device) * gate,
        features.graph.to(device) * (1.0 - gate),
        features.sim.to(device),
    )
    probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predictions = logits.argmax(1).cpu().numpy()
    truth = features.labels.numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(truth, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(truth, probabilities) if len(set(truth)) > 1 else float("nan"),
        "report": classification_report(
            truth, predictions, target_names=["Fake", "Real"], zero_division=0, output_dict=True
        ),
        "gates": features.gate.squeeze(1).numpy(),
        "labels": truth,
        "predictions": predictions,
        "probabilities": probabilities,
    }


@torch.no_grad()
def run_ablation(
    cfg: PipelineConfig, features: Features, fusion: AdaptiveFusionClassifier
) -> pd.DataFrame:
    device = cfg.torch_device
    content = features.content.to(device)
    evidence = features.evidence.to(device)
    graph = features.graph.to(device)
    sim = features.sim.to(device)
    gate = features.gate.to(device)
    truth = features.labels.numpy()

    variants = [
        ("Content + Evidence", True, False, True),
        ("Content + Graph", True, True, False),
        ("Content + Evidence + Graph (fixed 0.5)", False, True, True),
        ("TrustGraph-X (adaptive gate)", True, True, True),
    ]
    rows = []
    for name, use_gate, use_graph, use_evidence in variants:
        alpha = gate if use_gate else torch.full_like(gate, 0.5)
        e = evidence * alpha if use_evidence else torch.zeros_like(evidence)
        g = graph * (1.0 - alpha) if use_graph else torch.zeros_like(graph)
        predictions = fusion(content, e, g, sim).argmax(1).cpu().numpy()
        _, _, f1, _ = precision_recall_fscore_support(
            truth, predictions, average="binary", zero_division=0
        )
        rows.append({"Model": name, "Accuracy": accuracy_score(truth, predictions), "F1": f1})
    return pd.DataFrame(rows)


def run_shap(
    cfg: PipelineConfig,
    train: Features,
    explain: Features,
    fusion: AdaptiveFusionClassifier,
) -> pd.DataFrame:
    """Group-level SHAP attribution over content / evidence / graph / consistency blocks."""
    import shap

    device = cfg.torch_device
    dim = train.content.shape[1]

    def flatten(features: Features, count: int) -> np.ndarray:
        return torch.cat(
            [
                features.content[:count],
                features.evidence[:count],
                features.graph[:count],
                features.sim[:count],
            ],
            dim=1,
        ).numpy()

    def predict(matrix: np.ndarray) -> np.ndarray:
        x = torch.tensor(matrix, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = fusion(
                x[:, :dim], x[:, dim : 2 * dim] * 0.5, x[:, 2 * dim : 3 * dim] * 0.5, x[:, 3 * dim :]
            )
        return torch.softmax(logits, dim=1).cpu().numpy()[:, 1]

    background = flatten(train, min(20, len(train.labels)))
    samples = flatten(explain, min(cfg.shap_samples, len(explain.labels)))
    values = shap.KernelExplainer(predict, background).shap_values(samples, nsamples=100, silent=True)
    values = np.asarray(values).reshape(len(samples), -1)

    blocks = {
        "Content": slice(0, dim),
        "Evidence": slice(dim, 2 * dim),
        "Graph": slice(2 * dim, 3 * dim),
        "Consistency (cos CE, EG)": slice(3 * dim, 3 * dim + 2),
    }
    return pd.DataFrame(
        [
            {"Feature group": name, "Mean |SHAP|": float(np.abs(values[:, sl]).sum(axis=1).mean())}
            for name, sl in blocks.items()
        ]
    ).sort_values("Mean |SHAP|", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    config: PipelineConfig
    dataset_summary: pd.DataFrame
    split_summary: pd.DataFrame
    evidence_summary: dict[str, object]
    graph_summary: dict[str, object]
    contrastive_losses: list[float]
    fusion_history: pd.DataFrame
    metrics: dict[str, object]
    gate_analysis: pd.DataFrame
    ablation: pd.DataFrame
    shap_summary: pd.DataFrame | None
    encoder: TextEncoder
    bank: EvidenceBank
    graph: SemanticGraph
    model: TrustGraphX
    fusion: AdaptiveFusionClassifier


def run_pipeline(cfg: PipelineConfig, progress: Progress = _noop) -> PipelineResult:
    set_seed(cfg.seed)
    progress("Discovering datasets", 0.0)

    found = discover_datasets(cfg)
    standardized: dict[str, pd.DataFrame] = {}
    rows = []
    for name in LABELLED_SOURCES:
        path = found.get(name)
        if path is None:
            rows.append({"dataset": name, "file": "missing", "records": 0, "status": "skipped"})
            continue
        try:
            frame = standardize(load_raw(path), name, cfg)
            standardized[name] = frame
            rows.append(
                {
                    "dataset": name,
                    "file": path.name,
                    "records": len(frame),
                    "status": f"fake={int((frame.label == 0).sum())}, real={int((frame.label == 1).sum())}",
                }
            )
        except Exception as exc:  # noqa: BLE001 - reported back to the UI
            rows.append({"dataset": name, "file": path.name, "records": 0, "status": f"error: {exc}"})

    if not standardized:
        raise RuntimeError(
            f"No usable labelled dataset found in {cfg.data_dir}. Expected any of: "
            + ", ".join(cfg.files[name] for name in LABELLED_SOURCES)
        )

    master = build_master(cfg, standardized)
    if master["label"].nunique() < 2:
        raise RuntimeError("The combined dataset contains a single class; cannot train.")
    train_df, val_df, test_df = split_master(cfg, master)

    val_df = val_df.sample(min(cfg.val_limit, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
    test_df = test_df.sample(min(cfg.test_limit, len(test_df)), random_state=cfg.seed).reset_index(drop=True)
    train_small = train_df.sample(min(cfg.train_limit, len(train_df)), random_state=cfg.seed).reset_index(drop=True)

    split_summary = pd.DataFrame(
        [
            {"split": name, "records": len(frame), "fake": int((frame.label == 0).sum()), "real": int((frame.label == 1).sum())}
            for name, frame in [("train", train_small), ("validation", val_df), ("test", test_df)]
        ]
    )

    progress("Loading DeBERTa", 0.0)
    encoder = TextEncoder(cfg)

    progress("Building evidence bank", 0.0)
    bank = EvidenceBank.build(cfg, train_df, encoder, progress)

    pheme_path = found.get("pheme")
    if pheme_path is not None:
        pheme = load_raw(pheme_path)
        text_col = find_col(pheme, TEXT_COLUMNS)
        if text_col is None:
            raise RuntimeError(f"{pheme_path.name}: no text column found for the semantic graph.")
        graph_df = pd.DataFrame({"text": pheme[text_col].map(clean_text)})
        graph_source = pheme_path.name
    else:
        graph_df = train_df[["text"]].copy()
        graph_source = "training texts (Pheme.csv not found)"
    graph_df = graph_df[graph_df["text"].str.len() >= 20]

    progress("Building semantic graph", 0.0)
    graph = SemanticGraph.build(cfg, graph_df, encoder, progress)

    progress("Contrastive alignment", 0.0)
    train_small_for_cl = train_small.sample(min(len(train_small), cfg.train_limit), random_state=cfg.seed)
    _, contrastive_losses = train_contrastive(cfg, encoder, bank, train_small_for_cl, progress)

    model = TrustGraphX(encoder.hidden_size).to(cfg.torch_device)
    train_features = make_features(cfg, train_small, encoder, bank, graph, model, progress, "Train features")
    val_features = make_features(cfg, val_df, encoder, bank, graph, model, progress, "Validation features")
    test_features = make_features(cfg, test_df, encoder, bank, graph, model, progress, "Test features")

    progress("Training adaptive fusion", 0.0)
    fusion, history = train_fusion(cfg, train_features, val_features, progress)

    progress("Evaluating", 0.0)
    metrics = evaluate(cfg, test_features, fusion)
    gate_analysis = (
        pd.DataFrame({"label": metrics["labels"], "evidence_gate": metrics["gates"]})
        .groupby("label")["evidence_gate"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .replace({"label": {0: "Fake", 1: "Real"}})
    )
    ablation = run_ablation(cfg, test_features, fusion)

    shap_summary = None
    if cfg.run_shap:
        progress("Computing SHAP attribution", 0.0)
        shap_summary = run_shap(cfg, train_features, val_features, fusion)

    progress("Done", 1.0)
    return PipelineResult(
        config=cfg,
        dataset_summary=pd.DataFrame(rows),
        split_summary=split_summary,
        evidence_summary={
            "records": len(bank.frame),
            "sources": bank.frame["source"].value_counts().to_dict(),
        },
        graph_summary={
            "nodes": len(graph.texts),
            "edges": len(graph.texts) * cfg.graph_k,
            "source": graph_source,
            "mean_neighbor_similarity": graph.mean_neighbor_similarity,
        },
        contrastive_losses=contrastive_losses,
        fusion_history=history,
        metrics=metrics,
        gate_analysis=gate_analysis,
        ablation=ablation,
        shap_summary=shap_summary,
        encoder=encoder,
        bank=bank,
        graph=graph,
        model=model,
        fusion=fusion,
    )


@torch.no_grad()
def predict_claim(result: PipelineResult, text: str) -> dict[str, object]:
    cfg = result.config
    device = cfg.torch_device
    claim = clean_text(text)

    indices, scores = result.bank.retrieve(result.encoder, [claim], top_k=cfg.evidence_top_k)
    evidence_rows = [result.bank.frame.iloc[int(i)] for i in indices[0]]
    top_evidence = evidence_rows[0]["text"]

    out = result.model(
        result.encoder.embed([claim]).to(device),
        result.encoder.embed([top_evidence]).to(device),
        result.graph.neighbors(result.encoder, [claim], device),
    )
    gate = out["gate"]
    logits = result.fusion(
        out["content"],
        out["evidence"] * gate,
        out["graph"] * (1.0 - gate),
        torch.cat([out["ce_similarity"], out["eg_similarity"]], dim=1),
    )
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return {
        "prediction": "Real" if int(probabilities.argmax()) == 1 else "Fake",
        "probability_real": float(probabilities[1]),
        "probability_fake": float(probabilities[0]),
        "evidence_gate": float(gate.item()),
        "content_evidence_similarity": float(out["ce_similarity"].item()),
        "evidence_graph_similarity": float(out["eg_similarity"].item()),
        "retrieved_evidence": pd.DataFrame(
            [
                {
                    "similarity": float(scores[0][rank]),
                    "label": "Real" if int(row["label"]) == 1 else "Fake",
                    "source": row["source"],
                    "text": row["text"][:400],
                }
                for rank, row in enumerate(evidence_rows)
            ]
        ),
    }


def save_artifacts(result: PipelineResult, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "trustgraphx_v2.pt"
    torch.save(
        {
            "trustgraphx": result.model.state_dict(),
            "fusion": result.fusion.state_dict(),
            "model_name": result.config.model_name,
            "hidden_size": result.encoder.hidden_size,
            "graph_k": result.config.graph_k,
        },
        checkpoint,
    )
    bank_csv = out_dir / "training_only_evidence_bank.csv"
    result.bank.frame.to_csv(bank_csv, index=False)
    return [checkpoint, bank_csv]
