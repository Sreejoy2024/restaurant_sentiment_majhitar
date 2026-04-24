"""
Restaurant Recommendation Engine
Uses TF-IDF similarity + sentiment scores + food tags to suggest
restaurants based on natural language queries.
Example: "I want something cheesy and dessert type" → suggests matching restaurants.

Majhitar, Sikkim - NLP Assignment
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────
# FOOD PREFERENCE TAG VOCABULARY
# Maps user query keywords to restaurant tags
# ──────────────────────────────────────────────

PREFERENCE_MAPPING = {
    # Food types
    "cheesy": ["cheesy", "pasta", "pizza", "continental", "fine_dining"],
    "cheese": ["cheesy", "pasta", "pizza", "continental"],
    "dessert": ["desserts", "cafe", "cheesy", "ice_cream", "sweet"],
    "desserts": ["desserts", "cafe", "ice_cream", "sweet"],
    "sweet": ["desserts", "ice_cream", "cafe"],
    "ice cream": ["ice_cream", "desserts", "fast_food"],
    "cake": ["desserts", "cafe", "bakery"],
    "pastry": ["desserts", "cafe", "bakery"],
    "momo": ["local_cuisine", "tibetan", "authentic"],
    "momos": ["local_cuisine", "tibetan", "authentic"],
    "thukpa": ["local_cuisine", "tibetan", "authentic"],
    "biryani": ["indian", "north_indian", "spicy"],
    "tandoori": ["indian", "north_indian", "non_veg", "spicy"],
    "chicken": ["non_veg", "grilled", "indian"],
    "grilled": ["grilled", "non_veg", "bar"],
    "spicy": ["spicy", "indian", "north_indian"],
    "veg": ["vegetarian_friendly"],
    "vegetarian": ["vegetarian_friendly"],
    "pizza": ["fast_food", "cheesy", "continental"],
    "burger": ["fast_food", "continental"],
    "pasta": ["cheesy", "continental", "cafe"],
    "coffee": ["cafe", "drinks"],
    "drinks": ["bar", "cafe", "drinks"],
    "alcohol": ["bar", "late_night"],
    "bar": ["bar", "late_night"],
    "bbq": ["grilled", "bar", "non_veg"],
    "breakfast": ["breakfast", "budget"],
    "budget": ["budget", "cheap"],
    "cheap": ["budget", "cheap"],
    "affordable": ["budget", "cheap"],
    "fine dining": ["fine_dining", "premium", "scenic_view"],
    "fancy": ["fine_dining", "premium", "scenic_view"],
    "romantic": ["romantic", "scenic_view", "fine_dining"],
    "scenic": ["scenic_view", "river_view", "romantic"],
    "view": ["scenic_view", "river_view"],
    "local": ["local_cuisine", "authentic", "sikkimese"],
    "authentic": ["authentic", "local_cuisine"],
    "sikkimese": ["local_cuisine", "authentic"],
    "north indian": ["north_indian", "spicy", "indian"],
    "punjabi": ["north_indian", "spicy", "indian"],
    "chinese": ["chinese", "noodles"],
    "fast food": ["fast_food", "snacks"],
    "snacks": ["snacks", "fast_food", "quick_bite"],
    "quick": ["quick_bite", "quick_service", "fast_food"],
    "family": ["family_friendly"],
    "kids": ["family_friendly", "desserts"],
    "buffet": ["buffet", "family_friendly"],
    "traveler": ["traveler_friendly", "quick_bite"],
    "late night": ["late_night", "bar"],
}


class RestaurantRecommender:
    """
    NLP-powered restaurant recommender that:
    1. Parses user's natural language food preference query
    2. Matches against restaurant tags and cuisine data
    3. Boosts by average sentiment score from reviews
    4. Returns ranked list of suitable restaurants
    """

    def __init__(self, restaurant_metadata: List[Dict], reviews_df: pd.DataFrame):
        self.restaurants = restaurant_metadata
        self.reviews_df = reviews_df
        self._build_restaurant_profiles()
        self._compute_sentiment_scores()
        self._build_tfidf_index()

    def _build_restaurant_profiles(self):
        """Create rich text profiles for each restaurant."""
        self.profiles = {}
        for r in self.restaurants:
            profile_text = " ".join([
                r["name"],
                " ".join(r["cuisine"]),
                " ".join(r["specialty"]),
                " ".join(r["tags"]),
                r["price_range"],
            ]).lower()
            self.profiles[r["name"]] = profile_text

    def _compute_sentiment_scores(self):
        """Compute per-restaurant sentiment statistics from reviews."""
        self.sentiment_stats = {}

        for r in self.restaurants:
            name = r["name"]
            r_reviews = self.reviews_df[self.reviews_df["restaurant_name"] == name]

            if len(r_reviews) == 0:
                self.sentiment_stats[name] = {
                    "positive_pct": 0.5,
                    "avg_rating": 3.5,
                    "review_count": 0,
                    "sentiment_score": 0.5,
                }
                continue

            pos_pct = (r_reviews["sentiment_label"] == "positive").mean()
            avg_rating = r_reviews["rating"].mean()
            review_count = len(r_reviews)

            # Composite sentiment score (0-1)
            sentiment_score = (pos_pct * 0.6) + ((avg_rating / 5.0) * 0.4)

            self.sentiment_stats[name] = {
                "positive_pct": round(pos_pct, 3),
                "avg_rating": round(avg_rating, 2),
                "review_count": review_count,
                "sentiment_score": round(sentiment_score, 3),
            }

    def _build_tfidf_index(self):
        """Build TF-IDF index over restaurant profiles."""
        self.restaurant_names = list(self.profiles.keys())
        profile_texts = [self.profiles[n] for n in self.restaurant_names]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            analyzer="word",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(profile_texts)

    def _extract_tags_from_query(self, query: str) -> List[str]:
        """Map user query words to restaurant tags."""
        query_lower = query.lower()
        matched_tags = []
        for keyword, tags in PREFERENCE_MAPPING.items():
            if keyword in query_lower:
                matched_tags.extend(tags)
        return list(set(matched_tags))

    def _build_query_profile(self, query: str, tags: List[str]) -> str:
        """Combine query text and extracted tags into a search profile."""
        return (query.lower() + " " + " ".join(tags)).strip()

    def recommend(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Main recommendation method.
        
        Args:
            query: Natural language food preference (e.g., "cheesy dessert food")
            top_k: Number of restaurants to return
            
        Returns:
            List of restaurant recommendation dicts with scores and explanations
        """
        # Extract preference tags
        tags = self._extract_tags_from_query(query)

        # Build enriched query profile
        query_profile = self._build_query_profile(query, tags)

        # TF-IDF similarity
        query_vec = self.vectorizer.transform([query_profile])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Tag overlap score (direct matching)
        tag_scores = []
        for name in self.restaurant_names:
            r_meta = next((r for r in self.restaurants if r["name"] == name), None)
            if r_meta:
                r_tags = set(r_meta["tags"])
                overlap = len(set(tags) & r_tags) / max(len(tags), 1)
                tag_scores.append(overlap)
            else:
                tag_scores.append(0.0)
        tag_scores = np.array(tag_scores)

        # Sentiment boost
        sentiment_scores = np.array([
            self.sentiment_stats.get(name, {}).get("sentiment_score", 0.5)
            for name in self.restaurant_names
        ])

        # Combined score: TF-IDF (40%) + tag overlap (40%) + sentiment (20%)
        combined = (0.40 * similarities) + (0.40 * tag_scores) + (0.20 * sentiment_scores)

        # Rank and return top_k
        top_indices = np.argsort(combined)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            name = self.restaurant_names[idx]
            r_meta = next((r for r in self.restaurants if r["name"] == name), None)
            stats = self.sentiment_stats.get(name, {})

            # Build tag match explanation
            matched = set(tags) & set(r_meta["tags"]) if r_meta else set()
            why_list = []
            if matched:
                why_list.append(f"Matches your preference: {', '.join(matched)}")
            if stats.get("positive_pct", 0) > 0.6:
                why_list.append(f"{int(stats['positive_pct']*100)}% positive reviews")
            if r_meta:
                why_list.append(f"Specializes in: {', '.join(r_meta['specialty'][:3])}")

            recommendations.append({
                "rank": len(recommendations) + 1,
                "restaurant_name": name,
                "address": r_meta["address"] if r_meta else "Majhitar, Sikkim",
                "cuisine": ", ".join(r_meta["cuisine"]) if r_meta else "Multi-cuisine",
                "specialty": r_meta["specialty"] if r_meta else [],
                "price_range": r_meta["price_range"] if r_meta else "unknown",
                "avg_rating": stats.get("avg_rating", 0),
                "positive_pct": stats.get("positive_pct", 0),
                "review_count": stats.get("review_count", 0),
                "match_score": round(float(combined[idx]), 4),
                "tfidf_score": round(float(similarities[idx]), 4),
                "tag_score": round(float(tag_scores[idx]), 4),
                "sentiment_boost": round(float(sentiment_scores[idx]), 4),
                "why": why_list,
                "tags": r_meta["tags"] if r_meta else [],
            })

        return recommendations

    def get_restaurant_summary(self, restaurant_name: str) -> Dict:
        """Return full summary for a specific restaurant."""
        r_meta = next((r for r in self.restaurants if r["name"] == restaurant_name), None)
        stats = self.sentiment_stats.get(restaurant_name, {})
        r_reviews = self.reviews_df[self.reviews_df["restaurant_name"] == restaurant_name]

        if r_meta is None:
            return {"error": f"Restaurant '{restaurant_name}' not found."}

        # Top positive and negative reviews
        pos_reviews = r_reviews[r_reviews["sentiment_label"] == "positive"]["review_text"].head(3).tolist()
        neg_reviews = r_reviews[r_reviews["sentiment_label"] == "negative"]["review_text"].head(2).tolist()

        return {
            "name": restaurant_name,
            "address": r_meta["address"],
            "cuisine": r_meta["cuisine"],
            "specialty": r_meta["specialty"],
            "price_range": r_meta["price_range"],
            "tags": r_meta["tags"],
            "stats": stats,
            "top_positive_reviews": pos_reviews,
            "top_negative_reviews": neg_reviews,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.create_dataset import generate_reviews

    df, restaurants = generate_reviews()

    recommender = RestaurantRecommender(restaurants, df)

    queries = [
        "I want something cheesy and dessert type food",
        "Looking for spicy north Indian food, budget friendly",
        "Want a romantic scenic dinner place",
        "Quick momos and local Sikkimese food",
        "Bar and grilled food for late night",
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        recs = recommender.recommend(q, top_k=2)
        for r in recs:
            print(f"  [{r['rank']}] {r['restaurant_name']} | Score: {r['match_score']} | Rating: {r['avg_rating']}")
            print(f"       Why: {' | '.join(r['why'])}")
