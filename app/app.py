"""
Streamlit Web Application
Restaurant Sentiment Analysis & Recommendation System
Majhitar, Sikkim

Run: streamlit run app/app.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Path setup ──
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, ROOT_DIR)

# ── Page config ──
st.set_page_config(
    page_title="Majhitar Restaurant Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-title {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .hero-subtitle {
        color: #a8b2d8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a8b2d8;
        margin-top: 0.2rem;
    }
    .sentiment-positive {
        background: rgba(46,204,113,0.15);
        border-left: 4px solid #2ecc71;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .sentiment-negative {
        background: rgba(231,76,60,0.15);
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .sentiment-neutral {
        background: rgba(243,156,18,0.15);
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .rec-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 1px solid rgba(233,69,96,0.4);
        border-radius: 14px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        color: white;
    }
    .rec-rank { color: #e94560; font-size: 1.3rem; font-weight: 700; }
    .rec-name { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
    .rec-detail { color: #a8b2d8; font-size: 0.85rem; }
    .why-badge {
        background: rgba(233,69,96,0.2);
        border: 1px solid rgba(233,69,96,0.5);
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        color: #e94560;
        margin: 0.2rem;
        display: inline-block;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #c0392b, #e94560);
    }
    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data & Models ──
@st.cache_resource
def load_resources():
    data_path = os.path.join(ROOT_DIR, "data", "processed_reviews.csv")
    meta_path = os.path.join(ROOT_DIR, "data", "restaurant_metadata.json")
    rec_path = os.path.join(ROOT_DIR, "models", "recommender.pkl")

    # Load reviews
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Auto-generate if not present
        st.warning("Processed data not found. Running pipeline first...")
        from data.create_dataset import generate_reviews, save_restaurant_metadata
        from src.preprocessing import preprocess_pipeline
        os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)
        df_raw, restaurants = generate_reviews()
        df = preprocess_pipeline(df_raw)
        df.to_csv(data_path, exist_ok=True)

    # Load restaurant metadata
    import json
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            restaurants = json.load(f)
    else:
        from data.create_dataset import RESTAURANTS
        restaurants = RESTAURANTS

    # Load recommender
    if os.path.exists(rec_path):
        with open(rec_path, "rb") as f:
            recommender = pickle.load(f)
    else:
        from src.recommender import RestaurantRecommender
        recommender = RestaurantRecommender(restaurants, df)

    # Load VADER & TextBlob
    from src.sentiment_analyzer import VADERSentimentAnalyzer, TextBlobSentimentAnalyzer
    vader = VADERSentimentAnalyzer()
    textblob = TextBlobSentimentAnalyzer()

    return df, restaurants, recommender, vader, textblob


df, restaurants, recommender, vader_analyzer, textblob_analyzer = load_resources()


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍽️ Navigation")
    page = st.radio("", [
        "🏠 Dashboard",
        "🔍 Analyze Review",
        "🤖 Get Recommendation",
        "📊 Restaurant Insights",
        "📈 Model Evaluation",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.markdown(f"- 📝 **{len(df):,}** reviews")
    st.markdown(f"- 🏪 **{df['restaurant_name'].nunique()}** restaurants")
    pos_pct = (df["sentiment_label"] == "positive").mean() * 100
    st.markdown(f"- 😊 **{pos_pct:.0f}%** positive")

    st.markdown("---")
    st.markdown("**About**")
    st.caption("NLP Assignment — Text Analytics\nSentiment Analysis of Restaurant\nReviews in Majhitar, Sikkim")


# ──────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🍽️ Majhitar Restaurant Sentiment Analyzer</div>
  <div class="hero-subtitle">NLP-powered Sentiment Analysis & Smart Restaurant Recommendation · Majhitar, Sikkim</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────────────────────
if "Dashboard" in page:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (str(len(df)), "Total Reviews"),
        (str(df["restaurant_name"].nunique()), "Restaurants"),
        (f"{pos_pct:.0f}%", "Positive Reviews"),
        (f"{df['rating'].mean():.1f}★", "Avg Rating"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 Sentiment Distribution")
        counts = df["sentiment_label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
        wedge_colors = [colors.get(l, "#95a5a6") for l in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, colors=wedge_colors,
            autopct="%1.1f%%", startangle=140, pctdistance=0.82,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color("white")
            t.set_fontsize(10)
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("⭐ Average Rating by Restaurant")
        avg_r = df.groupby("restaurant_name")["rating"].mean().sort_values(ascending=True)
        short = [n[:18] + ".." if len(n) > 18 else n for n in avg_r.index]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
        bars = ax.barh(short, avg_r.values,
                       color=[plt.cm.RdYlGn(v / 5.0) for v in avg_r.values],
                       edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, avg_r.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8, color="white")
        ax.set_xlim(0, 5.5)
        ax.set_xlabel("Average Rating", color="white")
        ax.tick_params(colors="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📋 Recent Reviews Sample")
    sample = df[["restaurant_name", "review_text", "rating", "sentiment_label"]].sample(8, random_state=1)
    sample["sentiment_label"] = sample["sentiment_label"].map({
        "positive": "✅ Positive",
        "neutral": "⚪ Neutral",
        "negative": "❌ Negative",
    })
    st.dataframe(sample.rename(columns={
        "restaurant_name": "Restaurant",
        "review_text": "Review",
        "rating": "Stars",
        "sentiment_label": "Sentiment",
    }), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# PAGE: ANALYZE REVIEW
# ──────────────────────────────────────────────────────────────
elif "Analyze" in page:
    st.subheader("🔍 Analyze a Restaurant Review")
    st.caption("Enter any review text and get multi-model sentiment analysis instantly.")

    user_review = st.text_area(
        "Type or paste a review:",
        placeholder="e.g., The momos here were absolutely delicious! Best in Majhitar. Definitely coming back.",
        height=130,
    )

    if st.button("Analyze Sentiment"):
        if not user_review.strip():
            st.warning("Please enter a review to analyze.")
        else:
            v_res = vader_analyzer.predict_single(user_review)
            tb_res = textblob_analyzer.predict_single(user_review)

            # Determine final consensus
            votes = [v_res["label"], tb_res["label"]]
            from collections import Counter
            final = Counter(votes).most_common(1)[0][0]

            col1, col2, col3 = st.columns(3)
            for col, (model, label, score) in zip(
                [col1, col2, col3],
                [
                    ("VADER", v_res["label"], v_res["compound"]),
                    ("TextBlob", tb_res["label"], tb_res["polarity"]),
                    ("Consensus", final, None),
                ]
            ):
                icon = {"positive": "😊", "neutral": "😐", "negative": "😠"}.get(label, "❓")
                col.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem">{icon}</div>
                    <div class="metric-value" style="font-size:1.2rem">{label.upper()}</div>
                    <div class="metric-label">{model}</div>
                    {"<div class='metric-label'>Score: " + f"{score:.3f}" + "</div>" if score is not None else ""}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**VADER Details**")
                fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="none")
                cats = ["Positive", "Neutral", "Negative"]
                vals = [v_res["pos"], v_res["neu"], v_res["neg"]]
                bars = ax.bar(cats, vals,
                              color=["#2ecc71", "#f39c12", "#e74c3c"],
                              edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{val:.3f}",
                            ha="center", fontsize=9, color="white", fontweight="bold")
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score", color="white")
                ax.tick_params(colors="white")
                ax.set_facecolor("none")
                fig.patch.set_alpha(0)
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                st.markdown("**TextBlob Details**")
                fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="none")
                cats = ["Polarity", "Subjectivity"]
                vals = [tb_res["polarity"], tb_res["subjectivity"]]
                colors = ["#3498db" if cats[i] == "Polarity" else "#9b59b6" for i in range(len(cats))]
                bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01, f"{val:.3f}",
                            ha="center", fontsize=9, color="white", fontweight="bold")
                ax.set_ylim(-0.1, 1.2)
                ax.set_ylabel("Score", color="white")
                ax.tick_params(colors="white")
                ax.set_facecolor("none")
                fig.patch.set_alpha(0)
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                st.pyplot(fig)
                plt.close(fig)


# ──────────────────────────────────────────────────────────────
# PAGE: GET RECOMMENDATION
# ──────────────────────────────────────────────────────────────
elif "Recommendation" in page:
    st.subheader("🤖 Smart Restaurant Recommendation")
    st.caption("Tell us what you're craving in natural language and we'll find the perfect match in Majhitar!")

    example_queries = [
        "I want something cheesy and dessert type food",
        "Looking for spicy north Indian food, budget friendly",
        "Romantic scenic dinner place with good ambiance",
        "Quick momos and local Sikkimese cuisine",
        "Bar with grilled food for late night",
        "Family friendly place with buffet and vegetarian options",
    ]

    st.markdown("**Try an example or type your own:**")
    selected_example = st.selectbox("Example queries:", ["— Type your own below —"] + example_queries)

    user_query = st.text_input(
        "Your food preference:",
        value="" if selected_example == "— Type your own below —" else selected_example,
        placeholder="e.g., I want something cheesy and dessert type food...",
    )

    top_k = st.slider("Number of recommendations:", min_value=1, max_value=5, value=3)

    if st.button("🔍 Find Restaurants"):
        if not user_query.strip():
            st.warning("Please enter your food preference.")
        else:
            with st.spinner("Finding the best restaurants for you..."):
                recs = recommender.recommend(user_query, top_k=top_k)

            st.markdown(f"### Top {len(recs)} Matches for: *'{user_query}'*")

            for r in recs:
                stars = "⭐" * round(r["avg_rating"])
                price_icon = {"budget": "💰", "mid-range": "💰💰", "premium": "💰💰💰"}.get(r["price_range"], "💰")

                why_html = "".join([f'<span class="why-badge">{w}</span>' for w in r["why"]])

                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start">
                        <div>
                            <span class="rec-rank">#{r['rank']}</span>
                            <span class="rec-name" style="margin-left:0.5rem">{r['restaurant_name']}</span>
                        </div>
                        <div style="text-align:right">
                            <div style="color:#f39c12;font-size:1rem">{stars}</div>
                            <div class="rec-detail">{r['avg_rating']:.1f}/5 · {r['review_count']} reviews</div>
                        </div>
                    </div>
                    <div class="rec-detail" style="margin-top:0.6rem">
                        📍 {r['address']}<br>
                        🍴 {r['cuisine']}<br>
                        {price_icon} {r['price_range'].capitalize()}<br>
                        ✨ Specialties: {', '.join(r['specialty'][:4])}
                    </div>
                    <div style="margin-top:0.8rem">
                        <div class="rec-detail" style="margin-bottom:0.3rem">Why we recommend this:</div>
                        {why_html}
                    </div>
                    <div style="margin-top:0.8rem; display:flex; gap:1rem">
                        <div class="rec-detail">Match Score: <b style="color:#e94560">{r['match_score']:.3f}</b></div>
                        <div class="rec-detail">Sentiment: <b style="color:#2ecc71">{int(r['positive_pct']*100)}% positive</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PAGE: RESTAURANT INSIGHTS
# ──────────────────────────────────────────────────────────────
elif "Insights" in page:
    st.subheader("📊 Restaurant Deep Insights")

    restaurant_names = sorted(df["restaurant_name"].unique())
    selected = st.selectbox("Select Restaurant:", restaurant_names)

    r_df = df[df["restaurant_name"] == selected]

    col1, col2, col3, col4 = st.columns(4)
    stats = [
        (str(len(r_df)), "Reviews"),
        (f"{r_df['rating'].mean():.1f}★", "Avg Rating"),
        (f"{(r_df['sentiment_label']=='positive').mean()*100:.0f}%", "Positive"),
        (f"{(r_df['sentiment_label']=='negative').mean()*100:.0f}%", "Negative"),
    ]
    for col, (v, l) in zip([col1, col2, col3, col4], stats):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{v}</div>
            <div class="metric-label">{l}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Rating Distribution**")
        rating_counts = r_df["rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="none")
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#27ae60"],
                      edgecolor="white", linewidth=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(int(bar.get_height())),
                    ha="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Star Rating", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**Sentiment Breakdown**")
        sent_counts = r_df["sentiment_label"].value_counts()
        colors = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
        c = [colors.get(l, "#95a5a6") for l in sent_counts.index]
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="none")
        wedges, texts, autotexts = ax.pie(
            sent_counts.values, labels=sent_counts.index, colors=c,
            autopct="%1.1f%%", startangle=140, pctdistance=0.82,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for t in texts + autotexts:
            t.set_color("white")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**✅ Top Positive Reviews**")
        for _, row in r_df[r_df["sentiment_label"] == "positive"].head(3).iterrows():
            st.markdown(f'<div class="sentiment-positive">⭐ {row["rating"]}/5 — {row["review_text"]}</div>',
                        unsafe_allow_html=True)
    with col_neg:
        st.markdown("**❌ Recent Negative Reviews**")
        for _, row in r_df[r_df["sentiment_label"] == "negative"].head(3).iterrows():
            st.markdown(f'<div class="sentiment-negative">⭐ {row["rating"]}/5 — {row["review_text"]}</div>',
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# PAGE: MODEL EVALUATION
# ──────────────────────────────────────────────────────────────
elif "Evaluation" in page:
    st.subheader("📈 Model Evaluation Results")

    eval_dir = os.path.join(ROOT_DIR, "evaluation")

    if os.path.exists(os.path.join(eval_dir, "model_comparison.png")):
        st.image(os.path.join(eval_dir, "model_comparison.png"),
                 caption="Model Performance Comparison", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(os.path.join(eval_dir, "sentiment_distribution.png")):
            st.image(os.path.join(eval_dir, "sentiment_distribution.png"),
                     caption="Sentiment Distribution", use_container_width=True)
    with col2:
        if os.path.exists(os.path.join(eval_dir, "rating_vs_sentiment.png")):
            st.image(os.path.join(eval_dir, "rating_vs_sentiment.png"),
                     caption="Rating vs Sentiment", use_container_width=True)

    if os.path.exists(os.path.join(eval_dir, "top_words_by_sentiment.png")):
        st.image(os.path.join(eval_dir, "top_words_by_sentiment.png"),
                 caption="Top Words by Sentiment", use_container_width=True)

    if os.path.exists(os.path.join(eval_dir, "cross_validation_scores.png")):
        st.image(os.path.join(eval_dir, "cross_validation_scores.png"),
                 caption="Cross-Validation F1 Scores", use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Full Evaluation Report")
    report_path = os.path.join(eval_dir, "evaluation_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.code(f.read(), language=None)
    else:
        st.info("Run `python train.py` first to generate evaluation results.")
        st.markdown("""
        **Expected Results (approximate):**

        | Model | Accuracy | F1 (Weighted) |
        |---|---|---|
        | VADER | ~0.72 | ~0.70 |
        | TextBlob | ~0.68 | ~0.66 |
        | Logistic Regression | ~0.88 | ~0.87 |
        | Naive Bayes | ~0.82 | ~0.81 |
        | Linear SVM | ~0.87 | ~0.86 |
        | **Ensemble** | ~**0.85** | ~**0.84** |
        """)
