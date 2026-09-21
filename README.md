# Study Material Recommender

A data science final project (CM3070 / CM3005 Data Science).
A website where learners find study material, powered by a **machine-learning recommendation engine**
(not just manual filtering). Users can browse/filter courses by topic and level, and get a personalised
**"Recommended for you"** list driven by the model.

## What gets graded
The **data science**: the recommendation models and how well they are evaluated.
The website (Streamlit) is the demo wrapper on top.

## Plan (three models, increasing sophistication)
1. **Baseline – Collaborative Filtering** (matrix factorisation / SVD) — clears the pass bar.
2. **Content-Based** (TF-IDF on course text) — recommends similar material; helps with new items.
3. **Hybrid** (combine the two) — the target for a strong (2:1+) grade.

Evaluated with: precision@k, recall@k, RMSE, AUC, and k-fold cross-validation,
each compared against the baseline.

## Project structure
```
StudyMaterialRecommender/
├── data/                       # dataset CSVs (download separately — see data/README.md)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb   # START HERE: explore data + first model
├── src/
│   └── recommender.py          # reusable model code
├── app.py                      # Streamlit website
├── requirements.txt
└── README.md
```

## Setup
```bash
# 1. (recommended) create an environment
conda create -n recsys python=3.11 -y
conda activate recsys

# 2. install libraries
pip install -r requirements.txt

# 3. download the dataset into data/  (see data/README.md)

# 4. explore + build the first model
jupyter notebook notebooks/01_eda_and_baseline.ipynb

# 5. run the website (later, once a model is saved)
streamlit run app.py
```

## Where I am in CM3070
- [x] Project proposal video submitted
- [ ] Literature review (Ch.2 of PPR)
- [ ] Baseline recommender (this repo) → feature prototype for the PPR
- [ ] Content-based + hybrid models
- [ ] Evaluation + Final Report
