# Data folder

Put your dataset CSV files in this folder.

## Recommended dataset

**Course Reviews on Coursera** (Kaggle):
https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera

It contains two files:
- `Coursera_courses.csv` — the course catalogue (name, institution, course_url, etc.)
- `Coursera_reviews.csv` — individual user reviews with a `rating` (1–5) and the `course_id` / `reviewers`

Why this dataset:
- The **reviews + ratings** give you user→item interactions for **collaborative filtering** (the ML core).
- The **course metadata** powers **content-based recommendations** and the **tag/keyword filtering** on your website.

## How to download
1. Make a free Kaggle account.
2. Open the link above and click **Download**.
3. Unzip the files into this `data/` folder.
4. Open `notebooks/01_eda_and_baseline.ipynb` and run it.

> Note: the `data/` CSVs are git-ignored so you don't commit large files.
