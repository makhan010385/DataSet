# Datasets

## Streamlit app

An interactive explorer for the datasets in this repo (preview, column stats, charts, filtering and CSV export):

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default it reads the datasets next to `app.py`. To use another folder, set the dataset folder in the sidebar or start the app with `DATASET_DIR`:

```powershell
$env:DATASET_DIR = "E:\2026\Lincoln Conference\Third Coference\Dataset"
streamlit run app.py
```

## TrustGraph-X v2 app (fake-news detection)

Local Streamlit implementation of the TrustGraph-X pipeline: multi-dataset loading and
label normalization, a training-only evidence bank (PolitiFact/Snopes), DeBERTa embeddings,
contrastive claim–evidence alignment, a semantic KNN graph (PHEME) with graph attention,
evidence/graph consistency scores, an adaptive evidence gate, adaptive fusion classification,
ablations and SHAP attribution.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run trustgraph_app.py
```

Set the dataset folder in the sidebar (e.g. `E:\2026\Lincoln Conference\Third Coference\Dataset`).
Expected files (searched recursively, case-insensitive): `gossipcop.csv`, `ISOT.csv`, `kaggle.csv`,
`PolitiFact.csv`, `snopeswithsum.csv`, `WELFake_Dataset.csv`, `Pheme.csv`. Missing labelled files are
skipped; if `Pheme.csv` is absent the graph is built from training texts. Everything runs on CPU;
start with the default limits and raise them once a run completes.

### Datasets collected from R packages
 - mlbench 
 - kernlab
 - klaR
 - car
 - reshape2
 - hflights
 - ISLR
 
### The original source repositories are:
 - ftp://ftp.ics.uci.edu/pub/machine-learning-databases
 - http://www.ics.uci.edu/~mlearn/MLRepository.html
 - http://kdd.ics.uci.edu
 - http://www.liacs.nl/~putten/library/cc2000/ (ticdata)

