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

