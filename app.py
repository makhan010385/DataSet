"""Streamlit explorer for the datasets stored in this repository."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent
SUPPORTED_SUFFIXES = (".csv", ".xlsx", ".xls", ".json", ".txt")
MAX_PREVIEW_ROWS = 200

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def list_datasets() -> list[str]:
    files = [
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=str.lower)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(name: str, nrows: int | None) -> pd.DataFrame:
    path = DATA_DIR / name
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        return df.head(nrows) if nrows else df
    if suffix == ".json":
        df = pd.read_json(path, lines=False)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, nrows=nrows, sep=None, engine="python")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude=np.number).columns.tolist()


def overview_tab(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    st.subheader("Preview")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Columns")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )
    st.dataframe(summary, use_container_width=True)

    if numeric_columns(df):
        st.subheader("Numeric summary")
        st.dataframe(df[numeric_columns(df)].describe().T, use_container_width=True)


def explore_tab(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    filtered = df.copy()

    cat_cols = [c for c in categorical_columns(df) if df[c].nunique() <= 50]
    chosen_cats = st.multiselect("Filter categorical columns", cat_cols)
    for col in chosen_cats:
        values = st.multiselect(
            f"{col}", sorted(df[col].dropna().unique().tolist()), key=f"cat_{col}"
        )
        if values:
            filtered = filtered[filtered[col].isin(values)]

    chosen_nums = st.multiselect("Filter numeric columns", numeric_columns(df))
    for col in chosen_nums:
        lo, hi = float(df[col].min()), float(df[col].max())
        if lo == hi:
            continue
        low, high = st.slider(f"{col}", lo, hi, (lo, hi), key=f"num_{col}")
        filtered = filtered[filtered[col].between(low, high)]

    st.caption(f"{len(filtered):,} of {len(df):,} rows match the filters")
    st.dataframe(filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    buffer = io.StringIO()
    filtered.to_csv(buffer, index=False)
    st.download_button(
        "Download filtered CSV",
        buffer.getvalue(),
        file_name="filtered.csv",
        mime="text/csv",
    )
    return filtered


def visualize_tab(df: pd.DataFrame) -> None:
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    chart = st.selectbox(
        "Chart type", ["Histogram", "Scatter", "Box", "Bar (counts)", "Correlation heatmap"]
    )

    if chart == "Histogram" and nums:
        col = st.selectbox("Column", nums)
        color = st.selectbox("Color by", ["None"] + [c for c in cats if df[c].nunique() <= 20])
        fig = px.histogram(df, x=col, color=None if color == "None" else color, nbins=40)
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Scatter" and len(nums) >= 2:
        x = st.selectbox("X", nums)
        y = st.selectbox("Y", [c for c in nums if c != x])
        color = st.selectbox("Color by", ["None"] + [c for c in cats if df[c].nunique() <= 20])
        fig = px.scatter(
            df.head(5000), x=x, y=y, color=None if color == "None" else color, opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Box" and nums:
        y = st.selectbox("Value", nums)
        group = st.selectbox("Group by", ["None"] + [c for c in cats if df[c].nunique() <= 20])
        fig = px.box(df, y=y, x=None if group == "None" else group)
        st.plotly_chart(fig, use_container_width=True)
    elif chart == "Bar (counts)" and cats:
        col = st.selectbox("Column", cats)
        counts = df[col].value_counts().head(30).reset_index()
        counts.columns = [col, "count"]
        st.plotly_chart(px.bar(counts, x=col, y="count"), use_container_width=True)
    elif chart == "Correlation heatmap" and len(nums) >= 2:
        corr = df[nums].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"),
            use_container_width=True,
        )
    else:
        st.info("The selected dataset does not have suitable columns for this chart.")


def model_tab(df: pd.DataFrame) -> None:
    st.caption("Trains a random forest baseline on the selected target column.")
    target = st.selectbox("Target column", df.columns.tolist())
    features = st.multiselect(
        "Feature columns",
        [c for c in df.columns if c != target],
        default=[c for c in df.columns if c != target][:10],
    )
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

    if not st.button("Train model") or not features:
        return

    data = df[features + [target]].dropna()
    if len(data) < 20:
        st.error("Not enough complete rows to train a model.")
        return

    X = pd.get_dummies(data[features], drop_first=True)
    y = data[target]
    is_classification = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    if is_classification:
        y = y.astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = (
        RandomForestClassifier(n_estimators=200, random_state=42)
        if is_classification
        else RandomForestRegressor(n_estimators=200, random_state=42)
    )
    with st.spinner("Training..."):
        model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if is_classification:
        st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
    else:
        c1, c2 = st.columns(2)
        c1.metric("R²", f"{r2_score(y_test, preds):.3f}")
        c2.metric("RMSE", f"{root_mean_squared_error(y_test, preds):.3f}")

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    st.plotly_chart(
        px.bar(importance, x="importance", y="feature", orientation="h"),
        use_container_width=True,
    )


def main() -> None:
    st.title("📊 Dataset Explorer")

    datasets = list_datasets()
    with st.sidebar:
        st.header("Data source")
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        name = st.selectbox(
            "…or pick a dataset from this repo",
            datasets,
            index=datasets.index("IRIS.csv") if "IRIS.csv" in datasets else 0,
        )
        limit_rows = st.checkbox("Limit rows (faster for large files)", value=True)
        nrows = st.number_input("Rows to load", 100, 1_000_000, 20_000, 100) if limit_rows else None

    if uploaded is not None:
        df = pd.read_csv(uploaded, nrows=nrows)
        st.caption(f"Loaded uploaded file `{uploaded.name}`")
    else:
        try:
            df = load_dataset(name, int(nrows) if nrows else None)
        except Exception as exc:  # noqa: BLE001 - surface parse errors in the UI
            st.error(f"Could not load `{name}`: {exc}")
            return
        st.caption(f"Loaded `{name}` from the repository")

    overview, explore, visualize, model = st.tabs(
        ["Overview", "Explore", "Visualize", "Quick model"]
    )
    with overview:
        overview_tab(df)
    with explore:
        filtered = explore_tab(df)
    with visualize:
        visualize_tab(filtered if len(filtered) else df)
    with model:
        model_tab(df)


if __name__ == "__main__":
    main()
