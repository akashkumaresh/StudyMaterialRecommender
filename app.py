"""
app.py — the Study Material Recommender website (Streamlit).

Run with:   streamlit run app.py
(from inside the StudyMaterialRecommender folder, with the `recsys` environment active)

A clean, card-based UI on top of the ML models in src/recommender.py.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.recommender import (  # noqa: E402
    load_data, build_ratings, split, CollaborativeModel, ContentModel,
    HybridModel, recommend,
)

st.set_page_config(page_title="Study Material Recommender", page_icon="📚", layout="wide")

# --------------------------------------------------------------------------- #
#  Styling                                                                     #
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1150px; }
  /* Hero header */
  .hero {
    background: linear-gradient(120deg, #6d5efc 0%, #4b9bff 100%);
    padding: 2.2rem 2.4rem; border-radius: 18px; color: white;
    box-shadow: 0 10px 30px rgba(80,90,200,0.25); margin-bottom: 1.6rem;
  }
  .hero h1 { font-size: 2.2rem; margin: 0 0 .4rem 0; font-weight: 800; letter-spacing: -.5px; }
  .hero p  { margin: 0; opacity: .92; font-size: 1.02rem; }
  /* Section titles */
  .sec { font-size: 1.35rem; font-weight: 700; margin: 1.6rem 0 .8rem 0; }
  /* Chips */
  .chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .4rem; }
  .chip {
    background: #eef1ff; color: #4338ca; border: 1px solid #dfe3ff;
    padding: .32rem .7rem; border-radius: 999px; font-size: .82rem; font-weight: 600;
  }
  /* Recommendation cards */
  .rec {
    display: flex; align-items: center; gap: 1rem;
    background: #ffffff; border: 1px solid #ececf3; border-radius: 14px;
    padding: .85rem 1.1rem; margin-bottom: .6rem;
    box-shadow: 0 2px 8px rgba(20,20,50,0.04); transition: all .15s ease;
  }
  .rec:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(20,20,50,0.10); }
  .rank {
    flex: 0 0 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(120deg,#6d5efc,#4b9bff); color: white;
    display: flex; align-items: center; justify-content: center; font-weight: 700;
  }
  .rec-body { flex: 1; }
  .rec-name { font-weight: 600; font-size: 1rem; color: #1f2340; margin-bottom: .35rem; }
  .bar { background: #eef0f6; height: 7px; border-radius: 6px; overflow: hidden; }
  .bar-fill { height: 7px; border-radius: 6px; background: linear-gradient(90deg,#6d5efc,#4b9bff); }
  .match { flex: 0 0 58px; text-align: right; font-weight: 700; color: #6d5efc; font-size: .95rem; }
  .stat { font-size: 1.6rem; font-weight: 800; color: #4338ca; }
  .stat-label { color: #8a8fa3; font-size: .82rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Load data and train models (cached)                                         #
# --------------------------------------------------------------------------- #
def find_data_dir() -> Path:
    candidates = [ROOT / "data",
                  Path("/Users/akashkumaresh/Claude/Projects/final project/StudyMaterialRecommender/data")]
    candidates += [p / "data" for p in [Path.cwd(), *Path.cwd().parents]]
    for d in candidates:
        if (d / "Coursera_reviews.csv").exists():
            return d
    raise FileNotFoundError("Could not find the data folder with Coursera_reviews.csv.")


@st.cache_resource(show_spinner="Loading data and training the models (first run only)…")
def setup():
    reviews, courses = load_data(find_data_dir())
    ratings = build_ratings(reviews)
    train_df, _ = split(ratings)
    cf = CollaborativeModel().fit(train_df)
    content = ContentModel().fit(courses, train_df)
    cat = courses.drop_duplicates("course_id").reset_index(drop=True)
    name_col = "name" if "name" in cat.columns else cat.columns[0]
    cid_to_name = dict(zip(cat["course_id"], cat[name_col]))
    all_items = list(cat["course_id"])
    n_users = ratings["user"].nunique()
    return train_df, cat, name_col, cid_to_name, all_items, cf, content, n_users


train_df, cat, name_col, cid_to_name, all_items, cf, content, n_users = setup()


# --------------------------------------------------------------------------- #
#  Hero header                                                                 #
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="hero">
  <h1>📚 Study Material Recommender</h1>
  <p>Personalised course recommendations, powered by collaborative, content-based and hybrid machine-learning models.</p>
</div>
""", unsafe_allow_html=True)

# quick stats row
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="stat">{len(all_items):,}</div><div class="stat-label">Courses</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat">{n_users:,}</div><div class="stat-label">Learners</div>', unsafe_allow_html=True)
c3.markdown('<div class="stat">3</div><div class="stat-label">ML models</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Sidebar controls                                                            #
# --------------------------------------------------------------------------- #
st.sidebar.header("⚙️ Settings")
users = train_df["user"].value_counts().head(200).index.tolist()
user = st.sidebar.selectbox("Choose a learner", users)
alpha = st.sidebar.slider("Hybrid weight α  (0 = content · 1 = collaborative)", 0.0, 1.0, 0.0, 0.25)
n = st.sidebar.slider("Number of recommendations", 5, 15, 8)
st.sidebar.caption("α = 0 (pure content-based) scored best in evaluation.")


# --------------------------------------------------------------------------- #
#  Recommended for you                                                         #
# --------------------------------------------------------------------------- #
liked = train_df[(train_df["user"] == user) & (train_df["rating"] >= 4)]["item"].tolist()
if liked:
    st.markdown('<div class="sec">👍 Because you enjoyed</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="chip">{cid_to_name.get(i, i)}</span>' for i in liked[:6])
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

st.markdown('<div class="sec">✨ Recommended for you</div>', unsafe_allow_html=True)
recs = recommend(HybridModel(cf, content, alpha=alpha).score, train_df, all_items, user, n=n)
top = max((s for _, s in recs), default=1) or 1
for rank, (item, score) in enumerate(recs, 1):
    pct = int(score / top * 100)
    st.markdown(f"""
    <div class="rec">
      <div class="rank">{rank}</div>
      <div class="rec-body">
        <div class="rec-name">{cid_to_name.get(item, item)}</div>
        <div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>
      </div>
      <div class="match">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Browse the catalogue                                                        #
# --------------------------------------------------------------------------- #
st.markdown('<div class="sec">🔎 Browse the catalogue</div>', unsafe_allow_html=True)
if "institution" in cat.columns:
    inst = st.selectbox("Filter by institution", ["All"] + sorted(cat["institution"].dropna().unique().tolist()))
    view = cat if inst == "All" else cat[cat["institution"] == inst]
else:
    view = cat
cols = [c for c in [name_col, "institution", "course_url"] if c in view.columns]
st.dataframe(view[cols].head(100), use_container_width=True, hide_index=True)

st.caption("Built with Streamlit · models from src/recommender.py")
