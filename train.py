"""
Main Training Pipeline
Runs the full sentiment analysis pipeline:
  1. Generate/load dataset
  2. Preprocess text
  3. Train all models
  4. Evaluate & generate visualizations
  5. Save processed data and models

Run: python train.py
Majhitar, Sikkim Restaurant Reviews - NLP Assignment
"""

import os
import sys
import pickle

# ── Add src to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.create_dataset import generate_reviews, save_restaurant_metadata
from src.preprocessing import preprocess_pipeline
from src.sentiment_analyzer import train_all_models
from src.recommender import RestaurantRecommender
from src.evaluation import run_all_visualizations


def main():
    print("\n" + "=" * 70)
    print("  RESTAURANT SENTIMENT ANALYSIS — MAJHITAR, SIKKIM")
    print("  NLP Assignment | Text Analytics")
    print("=" * 70)

    # ─────────────────────────────────────
    # STEP 1: Generate Dataset
    # ─────────────────────────────────────
    print("\n[STEP 1] Generating dataset...")
    os.makedirs("data", exist_ok=True)

    raw_csv = "data/majhitar_restaurant_reviews.csv"
    meta_json = "data/restaurant_metadata.json"

    df_raw, restaurants = generate_reviews()
    df_raw.to_csv(raw_csv, index=False)
    save_restaurant_metadata(restaurants, meta_json)
    print(f"  Total reviews: {len(df_raw)}")
    print(f"  Restaurants: {df_raw['restaurant_name'].nunique()}")

    # ─────────────────────────────────────
    # STEP 2: Preprocess
    # ─────────────────────────────────────
    print("\n[STEP 2] Preprocessing text...")
    df_processed = preprocess_pipeline(df_raw, text_col="review_text")

    processed_csv = "data/processed_reviews.csv"
    df_processed.to_csv(processed_csv, index=False)
    print(f"  Processed dataset saved to {processed_csv}")

    # ─────────────────────────────────────
    # STEP 3: Train Models
    # ─────────────────────────────────────
    print("\n[STEP 3] Training sentiment models...")
    results, trained_models, (X_train, X_test, y_train, y_test) = train_all_models(
        df_processed,
        text_col="cleaned_text",
        label_col="sentiment_label",
    )

    # ─────────────────────────────────────
    # STEP 4: Evaluation & Visualizations
    # ─────────────────────────────────────
    print("\n[STEP 4] Generating evaluation visualizations...")
    os.makedirs("evaluation", exist_ok=True)
    run_all_visualizations(df_processed, results, save_dir="evaluation")

    # ─────────────────────────────────────
    # STEP 5: Build Recommender
    # ─────────────────────────────────────
    print("\n[STEP 5] Building restaurant recommender...")
    recommender = RestaurantRecommender(restaurants, df_processed)

    os.makedirs("models", exist_ok=True)
    with open("models/recommender.pkl", "wb") as f:
        pickle.dump(recommender, f)
    print("  Recommender saved to models/recommender.pkl")

    # ─────────────────────────────────────
    # STEP 6: Demo Recommendations
    # ─────────────────────────────────────
    print("\n[STEP 6] Demo recommendations...")
    demo_queries = [
        "I want something cheesy and dessert type food",
        "Looking for spicy north Indian food, budget friendly",
        "Want a romantic scenic dinner place",
    ]
    for query in demo_queries:
        print(f"\n  Query: '{query}'")
        recs = recommender.recommend(query, top_k=2)
        for r in recs:
            print(f"    [{r['rank']}] {r['restaurant_name']} "
                  f"(Score: {r['match_score']:.3f}, Rating: {r['avg_rating']:.1f}★)")
            for why in r["why"]:
                print(f"         • {why}")

    # ─────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  Dataset: {len(df_processed)} reviews | "
          f"{df_processed['restaurant_name'].nunique()} restaurants")
    print(f"  Models trained: {len(results)}")
    best = max(results.items(), key=lambda x: x[1]["f1_weighted"])
    print(f"  Best model: {best[0]} (F1={best[1]['f1_weighted']:.4f}, "
          f"Accuracy={best[1]['accuracy']:.4f})")
    print(f"  Outputs: data/, models/, evaluation/")
    print("\n  Run the app: streamlit run app/app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
