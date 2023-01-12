# PolTrans-Comparative-Study

## Install

$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r mlenv-packages.txt

## Run Jupyter Notebooks

$ jupyter lab  

## Directory Structure

├── 00A-Data Exploration.ipynb
├── 00B-Data Figure.ipynb
├── 01A-Forecast-PM2.5_ML-All.ipynb
├── 01B-Forecast-PM2.5_ML-Data.ipynb
├── 01C-Forecast-PM2.5_ML-Data-Paper.ipynb
├── 01D-Forecast-PM2.5_ML-TaylorDiagram.ipynb
├── 02A-Forecast-PM2.5_Stat-All.ipynb
├── 02B-Forecast-PM2.5_Stat-Data.ipynb
├── 02C-Forecast-PM2.5_Stat-Data-Paper.ipynb
├── 02D-Forecast-PM2.5_Stat-TaylorDiagram.ipynb
├── 03A-Forecast-PM2.5_Transformer-All.ipynb
├── 03A-Forecast-PM2.5_Transformer-All.py
├── 03B-Forecast-PM2.5_Transformer-Data.ipynb
├── 03C-Forecast-PM2.5_Transformer-Data-Paper.ipynb
├── 03D-Forecast-PM2.5_Transformer-Data-LinePlot.ipynb
├── 04A-Forecast-PM2.5_DL-All.ipynb
├── 04B-Forecast-PM2.5_DL-Data.ipynb
├── 04C-Forecast-PM2.5_DL-Data-Paper.ipynb
├── 04D-Forecast-PM2.5_DL-TaylorDiagram.ipynb
├── A0-Transformer-Basic.ipynb
├── A1-Transformer Time-Series.ipynb
├── A2-Time2Vec.ipynb
├── DM001-AirQualityUCI.ipynb
├── DM002-Beijing.ipynb
├── DS001-Delhi Dataset Creation.ipynb
├── DS002-New York Dataset Creation.ipynb
├── DS003-Seoul Dataset Creation.ipynb
├── DS004-Ulaanbaatar Dataset Creation.ipynb
├── DS005-Skopje Dataset Creation.ipynb
├── datasets
│   ├── air_quality_uci_dataset.pkl
│   ├── beijing_pm10_dataset.pkl
│   ├── beijing_pm25_dataset.pkl
│   ├── delhi_dataset.pkl
│   ├── nyc_dataset.pkl
│   ├── seoul_dataset.pkl
│   ├── skopje_dataset.pkl
│   └── ulaanbaatar_dataset.pkl
├── README.txt
├── mlenv-packages.txt
├── taylor_diagram.py
└── tstransformer.py

Note
---
1. For reproducing results, follow the notebook sequence from 00A - 04D. 
2. DS notebook series are to understand how datasets were created.
3. DM series show data manipulation from original data. 
4. A series are practice environments for Transformer experimentation.
5. tstransformer.py is the python script housing the PolTrans architecture.
6. taylor_diagram.py contains custom code for the Taylor Diagrams supplied in the figures.