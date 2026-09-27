"""Streamlit explorer for the datasets stored in this repository."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent
SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".txt"}
MAX_PREVIEW_ROWS = 500

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def list_datasets() -> list[str]:
    files = [
        p
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(p.name for p in files)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(name: str, nrows: int | None) -> pd.DataFrame:
    path = DATA_DIR / name
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
        return df.head(nrows) if nrows else df
    if suffix == ".json":
        df = pd.read_json(path, lines=False)
        return df.head(nrows) if nrows else df
    sep = "\t" if suffix == ".txt" else None
    return pd.read_csv(path, sep=sep, engine="python", nrows=nrows)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def render_overview(df: pd.DataFrame, name: str) -> None:
    size_mb = (DATA_DIR / name).stat().st_size / 1024**2
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows loaded", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    col4.metric("File size", f"{size_mb:.2f} MB")

    st.subheader("Preview")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Columns")
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )
    st.dataframe(info, use_container_width=True)


def render_statistics(df: pd.DataFrame) -> None:
    nums = numeric_columns(df)
    cats = categorical_columns(df)

    if nums:
        st.subheader("Numeric summary")
        st.dataframe(df[nums].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns in this dataset.")

    if cats:
        st.subheader("Categorical summary")
        st.dataframe(df[cats].describe().T, use_container_width=True)

        column = st.selectbox("Value counts for column", cats)
        counts = df[column].value_counts(dropna=False).head(30).rename("count")
        st.dataframe(counts.to_frame(), use_container_width=True)

    if len(nums) >= 2:
        st.subheader("Correlation matrix")
        corr = df[nums].corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r"),
            use_container_width=True,
        )


def render_charts(df: pd.DataFrame) -> None:
    nums = numeric_columns(df)
    all_cols = df.columns.tolist()
    chart_type = st.selectbox(
        "Chart type", ["Histogram", "Scatter", "Line", "Bar", "Box"]
    )

    if chart_type == "Histogram":
        if not nums:
            st.info("Histogram needs a numeric column.")
            return
        column = st.selectbox("Column", nums)
        bins = st.slider("Bins", 5, 100, 30)
        fig = px.histogram(df, x=column, nbins=bins)
    elif chart_type == "Scatter":
        if len(nums) < 2:
            st.info("Scatter needs at least two numeric columns.")
            return
        x = st.selectbox("X axis", nums, index=0)
        y = st.selectbox("Y axis", nums, index=1)
        color = st.selectbox("Color by", ["(none)"] + all_cols)
        fig = px.scatter(df, x=x, y=y, color=None if color == "(none)" else color)
    elif chart_type == "Line":
        if not nums:
            st.info("Line chart needs a numeric column.")
            return
        x = st.selectbox("X axis", all_cols)
        y = st.multiselect("Y axis", nums, default=nums[:1])
        if not y:
            st.info("Pick at least one Y column.")
            return
        fig = px.line(df.sort_values(x), x=x, y=y)
    elif chart_type == "Bar":
        x = st.selectbox("Category", all_cols)
        y_options = ["(count)"] + nums
        y = st.selectbox("Value", y_options)
        if y == "(count)":
            data = df[x].value_counts().head(30).rename_axis(x).reset_index(name="count")
            fig = px.bar(data, x=x, y="count")
        else:
            data = df.groupby(x, dropna=False)[y].mean().head(30).reset_index()
            fig = px.bar(data, x=x, y=y)
    else:
        if not nums:
            st.info("Box plot needs a numeric column.")
            return
        y = st.selectbox("Value", nums)
        group = st.selectbox("Group by", ["(none)"] + categorical_columns(df))
        fig = px.box(df, y=y, x=None if group == "(none)" else group)

    st.plotly_chart(fig, use_container_width=True)


def render_filter(df: pd.DataFrame, name: str) -> None:
    filtered = df
    columns = st.multiselect("Filter on columns", df.columns.tolist())

    for column in columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and series.notna().any():
            low, high = float(series.min()), float(series.max())
            if low == high:
                st.caption(f"`{column}` has a single value ({low}); skipped.")
                continue
            selected = st.slider(column, low, high, (low, high))
            filtered = filtered[filtered[column].between(*selected)]
        else:
            options = series.dropna().astype(str).unique().tolist()[:1000]
            chosen = st.multiselect(f"{column} values", options)
            if chosen:
                filtered = filtered[filtered[column].astype(str).isin(chosen)]

    st.caption(f"{len(filtered):,} of {len(df):,} rows match.")
    st.dataframe(filtered.head(MAX_PREVIEW_ROWS), use_container_width=True)

    buffer = io.StringIO()
    filtered.to_csv(buffer, index=False)
    st.download_button(
        "Download filtered CSV",
        buffer.getvalue(),
        file_name=f"filtered_{Path(name).stem}.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("📊 Dataset Explorer")
    st.caption("Browse, summarise and visualise the datasets in this repository.")

    datasets = list_datasets()
    with st.sidebar:
        st.header("Dataset")
        query = st.text_input("Search", placeholder="e.g. iris")
        matches = [d for d in datasets if query.lower() in d.lower()] or datasets
        default = matches.index("Iris.csv") if "Iris.csv" in matches else 0
        name = st.selectbox("File", matches, index=default)
        limit_rows = st.checkbox("Limit rows while loading", value=True)
        nrows = st.number_input(
            "Max rows", min_value=100, max_value=1_000_000, value=5000, step=500
        ) if limit_rows else None
        st.caption(f"{len(datasets)} datasets available")

    try:
        df = load_dataset(name, int(nrows) if nrows else None)
    except Exception as exc:  # noqa: BLE001 - surface any parsing failure to the user
        st.error(f"Could not load `{name}`: {exc}")
        return

    if df.empty:
        st.warning(f"`{name}` loaded with no rows.")
        return

    overview, stats, charts, explore = st.tabs(
        ["Overview", "Statistics", "Charts", "Filter & export"]
    )
    with overview:
        render_overview(df, name)
    with stats:
        render_statistics(df)
    with charts:
        render_charts(df)
    with explore:
        render_filter(df, name)


if __name__ == "__main__":
    main()
