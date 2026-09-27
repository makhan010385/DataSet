# Datasets

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


## Streamlit Dataset Explorer

An interactive app for browsing the datasets in this repository.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tabs: Overview (shape, dtypes, missing values, summary stats), Explore (filters +
CSV download), Visualize (histogram, scatter, box, bar, correlation heatmap) and
Quick model (random forest baseline with metrics and feature importance).
