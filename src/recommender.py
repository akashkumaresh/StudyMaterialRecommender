"""
recommender.py
Core models and evaluation for the Study Material Recommender (CM3070 / CM3005).

This module consolidates the work developed across notebooks 01-05 into one
reusable, documented codebase. It provides:

  * Data loading and preprocessing.
  * Three recommendation models:
      - CollaborativeModel  (SVD matrix factorisation)
      - ContentModel        (TF-IDF course similarity)
      - HybridModel         (weighted blend of the two)
  * Evaluation utilities:
      - ground_truth / top-N recall@k / precision_recall_at_k
      - tune_alpha          (find the best hybrid weight)
      - significance_test   (paired t-test + Wilcoxon)

All model `score(user, item)` methods return a value normalised to [0, 1] so the
models can be compared and blended on a common scale.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD


# ===================================================================== #
#  Data                                                                 #
# ===================================================================== #
def load_data(data_dir: str | Path):
    """Load the reviews and courses CSVs from a data directory."""
    data_dir = Path(data_dir)
    reviews = pd.read_csv(data_dir / "Coursera_reviews.csv")
    courses = pd.read_csv(data_dir / "Coursera_courses.csv")
    return reviews, courses


def build_ratings(reviews: pd.DataFrame, min_reviews: int = 3) -> pd.DataFrame:
    """Reduce raw reviews to a clean (user, item, rating) table.

    Users with fewer than `min_reviews` ratings are dropped so collaborative
    filtering has enough signal per user.
    """
    df = reviews[["reviewers", "course_id", "rating"]].dropna()
    df.columns = ["user", "item", "rating"]
    active = df["user"].value_counts()
    df = df[df["user"].isin(active[active >= min_reviews].index)]
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42):
    """Train/test split of the ratings table."""
    return train_test_split(df, test_size=test_size, random_state=seed)


# ===================================================================== #
#  Model A: Collaborative filtering (SVD)                               #
# ===================================================================== #
class CollaborativeModel:
    """Matrix-factorisation collaborative filtering using Surprise SVD."""

    def __init__(self, rating_scale=(1, 5), seed: int = 42):
        self.rating_scale = rating_scale
        self.seed = seed
        self.svd = None

    def fit(self, train_df: pd.DataFrame) -> "CollaborativeModel":
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(train_df[["user", "item", "rating"]], reader)
        self.svd = SVD(random_state=self.seed)
        self.svd.fit(data.build_full_trainset())
        return self

    def score(self, user, item) -> float:
        """Predicted rating, normalised to [0, 1]."""
        lo, hi = self.rating_scale
        return (self.svd.predict(user, item).est - lo) / (hi - lo)


# ===================================================================== #
#  Model B: Content-based filtering (TF-IDF)                            #
# ===================================================================== #
class ContentModel:
    """Recommends courses whose text is similar to a user's liked courses."""

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold
        self.sim = None
        self.cid_to_idx = {}
        self.user_liked = defaultdict(list)

    def fit(self, courses: pd.DataFrame, train_df: pd.DataFrame) -> "ContentModel":
        cat = courses.drop_duplicates("course_id").reset_index(drop=True)
        name_col = "name" if "name" in cat.columns else cat.columns[0]
        text_cols = [c for c in [name_col, "institution", "skills", "description"]
                     if c in cat.columns]
        cat["text"] = cat[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

        tfidf = TfidfVectorizer(stop_words="english")
        self.sim = cosine_similarity(tfidf.fit_transform(cat["text"]))
        self.cid_to_idx = {c: i for i, c in enumerate(cat["course_id"])}

        self.user_liked = defaultdict(list)
        for u, it, r in train_df[["user", "item", "rating"]].itertuples(index=False):
            if r >= self.like_threshold and it in self.cid_to_idx:
                self.user_liked[u].append(self.cid_to_idx[it])
        return self

    def score(self, user, item) -> float:
        """Mean similarity between `item` and the user's liked courses, in [0, 1]."""
        liked = self.user_liked.get(user, [])
        j = self.cid_to_idx.get(item)
        if not liked or j is None:
            return 0.0
        return float(self.sim[j, liked].mean())


# ===================================================================== #
#  Model C: Hybrid                                                      #
# ===================================================================== #
class HybridModel:
    """Weighted blend: alpha * collaborative + (1 - alpha) * content."""

    def __init__(self, collaborative: CollaborativeModel,
                 content: ContentModel, alpha: float = 0.5):
        self.cf = collaborative
        self.content = content
        self.alpha = alpha

    def score(self, user, item) -> float:
        return self.alpha * self.cf.score(user, item) + \
               (1 - self.alpha) * self.content.score(user, item)


# ===================================================================== #
#  Recommendation                                                       #
# ===================================================================== #
def recommend(score_fn, train_df: pd.DataFrame, all_items, user, n: int = 10):
    """Top-N items (by score_fn) that the user has not already rated."""
    seen = set(train_df.loc[train_df["user"] == user, "item"])
    scored = [(it, score_fn(user, it)) for it in all_items if it not in seen]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


# ===================================================================== #
#  Evaluation                                                           #
# ===================================================================== #
def ground_truth(test_df: pd.DataFrame, threshold: float = 4.0):
    """For each user, the set of items they rated >= threshold in the test set."""
    pos = defaultdict(set)
    for u, it, r in test_df[["user", "item", "rating"]].itertuples(index=False):
        if r >= threshold:
            pos[u].add(it)
    return pos


def per_user_recall(score_fn, train_df, test_pos, all_items, users, k: int = 10):
    """Recall@k for each user: fraction of their liked items found in the top-k
    recommendations drawn from the whole catalogue."""
    out = {}
    for u in users:
        seen = set(train_df.loc[train_df["user"] == u, "item"])
        cand = [it for it in all_items if it not in seen]
        topk = sorted(cand, key=lambda it: score_fn(u, it), reverse=True)[:k]
        out[u] = len(set(topk) & test_pos[u]) / len(test_pos[u])
    return out


def evaluate_topn(score_fn, train_df, test_pos, all_items, users, k: int = 10):
    """Mean precision@k and recall@k over `users`."""
    precs, recs = [], []
    for u in users:
        seen = set(train_df.loc[train_df["user"] == u, "item"])
        cand = [it for it in all_items if it not in seen]
        topk = sorted(cand, key=lambda it: score_fn(u, it), reverse=True)[:k]
        hits = len(set(topk) & test_pos[u])
        precs.append(hits / k)
        recs.append(hits / len(test_pos[u]))
    return round(float(np.mean(precs)), 4), round(float(np.mean(recs)), 4)


def tune_alpha(cf, content, train_df, test_pos, all_items, users,
               alphas=(0.0, 0.25, 0.5, 0.75, 1.0), k: int = 10):
    """Sweep the hybrid weight and return a DataFrame of recall@k plus the best alpha."""
    rows = []
    for a in alphas:
        hyb = HybridModel(cf, content, alpha=a)
        pu = per_user_recall(hyb.score, train_df, test_pos, all_items, users, k)
        rows.append((a, round(float(np.mean(list(pu.values()))), 4)))
    table = pd.DataFrame(rows, columns=["alpha", f"recall@{k}"])
    best_alpha = table.loc[table[f"recall@{k}"].idxmax(), "alpha"]
    return table, best_alpha


def significance_test(recall_a: dict, recall_b: dict, users):
    """Paired t-test and Wilcoxon test comparing two models' per-user recall."""
    from scipy.stats import ttest_rel, wilcoxon
    a = np.array([recall_a[u] for u in users])
    b = np.array([recall_b[u] for u in users])
    result = {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
              "mean_improvement": float((b - a).mean())}
    result["t_p"] = float(ttest_rel(b, a).pvalue)
    try:
        result["wilcoxon_p"] = float(wilcoxon(b, a).pvalue)
    except ValueError:
        result["wilcoxon_p"] = None
    return result


# ===================================================================== #
#  Smoke test                                                           #
# ===================================================================== #
if __name__ == "__main__":
    reviews, courses = load_data("data")
    ratings = build_ratings(reviews)
    train_df, test_df = split(ratings)

    cf = CollaborativeModel().fit(train_df)
    content = ContentModel().fit(courses, train_df)
    all_items = list(courses["course_id"].drop_duplicates())

    test_pos = ground_truth(test_df)
    users = [u for u in test_pos if test_pos[u]][:300]

    table, best_alpha = tune_alpha(cf, content, train_df, test_pos, all_items, users)
    print(table.to_string(index=False))
    print("Best alpha:", best_alpha)

    sample = train_df["user"].iloc[0]
    print("\nTop recommendations for", sample)
    for item, sc in recommend(HybridModel(cf, content, best_alpha).score,
                              train_df, all_items, sample, n=10):
        print(f"  {sc:.3f}  {item}")
