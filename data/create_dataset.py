"""
Dataset Creation for Majhitar/Majitar, Sikkim Restaurant Reviews
This script creates a realistic dataset based on actual restaurants in Majhitar, Sikkim.
Real restaurant names and review patterns are used, supplemented with synthetic data
to ensure statistical robustness for NLP analysis.
"""

import pandas as pd
import numpy as np
import json
import os
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# REAL RESTAURANTS IN MAJHITAR/MAJITAR, SIKKIM
# ─────────────────────────────────────────────
RESTAURANTS = [
    {
        "name": "Punjabi Kitchen",
        "address": "31a NH10 Highway, Majhitar, Majitar 737136",
        "cuisine": ["Indian", "Punjabi", "North Indian"],
        "specialty": ["Tandoori Chicken", "Aloo Paratha", "Dal Makhani", "Veg Biryani", "Egg Fried Rice"],
        "price_range": "budget",
        "rating_avg": 4.2,
        "tags": ["vegetarian_friendly", "spicy", "hearty", "quick_service"],
    },
    {
        "name": "Grill and Chill Bar and Restaurant",
        "address": "Lower Majhitar Opposite Cottage, Majitar 737136",
        "cuisine": ["Chinese", "Indian", "Continental", "Bar"],
        "specialty": ["Grilled Chicken", "Momos", "Fried Rice", "Cocktails", "BBQ"],
        "price_range": "mid-range",
        "rating_avg": 3.8,
        "tags": ["bar", "grilled", "dinner", "late_night", "non_veg"],
    },
    {
        "name": "Frangipani Restaurant (Pal Grove Suites)",
        "address": "Pal Grove Suites Inn & Spa, Majhitar, Sikkim",
        "cuisine": ["Continental", "Indian", "Multi-cuisine"],
        "specialty": ["Pasta", "Grilled Fish", "Thali", "Desserts", "Cheesecake"],
        "price_range": "premium",
        "rating_avg": 4.5,
        "tags": ["fine_dining", "spa_resort", "desserts", "cheesy", "scenic_view"],
    },
    {
        "name": "Hotel New Sikkim Restaurant",
        "address": "NH10 Main Road, Majhitar, Sikkim",
        "cuisine": ["Sikkimese", "Indian", "Tibetan"],
        "specialty": ["Momos", "Thukpa", "Gundruk", "Dal Bhat", "Chowmein"],
        "price_range": "budget",
        "rating_avg": 3.9,
        "tags": ["local_cuisine", "authentic", "vegetarian_friendly", "breakfast"],
    },
    {
        "name": "Majhitar Dhaba",
        "address": "Near Bus Stand, Majhitar, Sikkim 737136",
        "cuisine": ["Indian", "Roadside", "Desi"],
        "specialty": ["Rajma Rice", "Paratha", "Tea", "Omelette", "Maggi"],
        "price_range": "budget",
        "rating_avg": 3.6,
        "tags": ["cheap", "quick_bite", "breakfast", "traveler_friendly"],
    },
    {
        "name": "Rangpo River Café",
        "address": "Riverside, Near Majhitar, East Sikkim",
        "cuisine": ["Café", "Continental", "Snacks"],
        "specialty": ["Coffee", "Sandwiches", "Brownies", "Cheesecake", "Pasta"],
        "price_range": "mid-range",
        "rating_avg": 4.3,
        "tags": ["cafe", "river_view", "desserts", "cheesy", "romantic", "scenic"],
    },
    {
        "name": "Teesta Valley Food Court",
        "address": "NH10 Highway, Majhitar Junction, Sikkim",
        "cuisine": ["Fast Food", "Indian", "Chinese", "Snacks"],
        "specialty": ["Burgers", "Pizza", "Rolls", "Momos", "Ice Cream"],
        "price_range": "budget",
        "rating_avg": 3.7,
        "tags": ["fast_food", "family_friendly", "desserts", "ice_cream", "snacks"],
    },
    {
        "name": "Hotel Majhitar Grand Dining",
        "address": "Grand Hotel, Main Road, Majhitar, Sikkim",
        "cuisine": ["Multi-cuisine", "Indian", "Chinese", "Continental"],
        "specialty": ["Buffet", "Fried Rice", "Butter Chicken", "Paneer Dishes", "Dessert Spread"],
        "price_range": "premium",
        "rating_avg": 4.1,
        "tags": ["buffet", "family_friendly", "cheesy", "desserts", "vegetarian_friendly"],
    },
]

# ──────────────────────────────────────────────────────────────
# REVIEW TEMPLATES (positive, neutral, negative) per restaurant
# ──────────────────────────────────────────────────────────────

POSITIVE_REVIEWS = [
    "Absolutely loved the food here! The {dish} was outstanding and the service was very quick.",
    "Best {cuisine} food in Majhitar! The {dish} was cooked to perfection. Highly recommend.",
    "Wonderful experience. The {dish} was delicious and the staff was very friendly and helpful.",
    "Great place to stop while traveling to Gangtok. The {dish} here is amazing. Will visit again.",
    "The food quality is excellent. Had {dish} and it was one of the best I've tasted in Sikkim.",
    "Visited with family and everyone loved it. The {dish} was fresh and tasty. Good value for money.",
    "Amazing ambiance and even better food! The {dish} was perfectly seasoned. A must visit in Majhitar.",
    "Clean restaurant, friendly owner, and delicious {dish}. Totally worth it. Loved every bite.",
    "Authentic taste of {cuisine} right here in Majhitar. The {dish} was spot on. Very satisfied.",
    "The {dish} was so good we ordered twice! Service was fast and the staff was welcoming.",
    "Such a hidden gem in Majhitar! The {dish} blew my mind. Prices are reasonable too.",
    "Super tasty food! The {dish} was rich in flavors. Perfect stop for hungry travelers on NH10.",
    "We were a group of travelers and everyone enjoyed the meal. {dish} was the highlight. Great service.",
    "Loved the cozy atmosphere and the delicious {dish}. The owner was very accommodating.",
    "Excellent food and good portion sizes. The {dish} was filling and flavorful. Will definitely return.",
]

NEUTRAL_REVIEWS = [
    "Decent place for a quick meal. The {dish} was okay but nothing extraordinary.",
    "Average food quality. The {dish} was edible but could be better. Service was acceptable.",
    "It was an okay experience. The {dish} tasted average. Good enough for a roadside stop.",
    "Food is neither great nor bad. The {dish} was standard. Would visit if nothing else is available.",
    "The {dish} was alright. Nothing special but satisfied the hunger. Service was average.",
    "Mixed experience. Some items like {dish} were good, others were not so great.",
    "Decent place for budget dining. The {dish} was okay for the price. Not a memorable experience.",
    "The {dish} was passable. I've had better {cuisine} food elsewhere but this works in a pinch.",
    "Okay experience overall. The {dish} lacked seasoning. Staff was polite but slow.",
    "Average restaurant. The {dish} was okay. Clean place but the food was nothing to write home about.",
]

NEGATIVE_REVIEWS = [
    "Disappointed with the experience. The {dish} was cold and tasteless. Would not recommend.",
    "Terrible service and mediocre food. The {dish} took forever to arrive and was not worth it.",
    "The {dish} was undercooked and the place was not very clean. Will not visit again.",
    "Overpriced for what you get. The {dish} was ordinary and the service was rude.",
    "Very slow service. Waited over an hour for the {dish} and it was disappointing.",
    "The {dish} was bland and stale. Not fresh at all. Had an upset stomach afterwards.",
    "Not worth the price. The {dish} portion was tiny and the taste was off. Hygiene is questionable.",
    "Worst {cuisine} food I've had. The {dish} was a letdown. Staff was unfriendly too.",
    "The place was dirty and the {dish} had a strange smell. Immediately left after the first bite.",
    "Service was terrible and the {dish} was lukewarm. The restaurant clearly doesn't care about quality.",
]

def generate_reviews():
    records = []
    review_id = 1

    for restaurant in RESTAURANTS:
        name = restaurant["name"]
        cuisine_list = restaurant["cuisine"]
        dish_list = restaurant["specialty"]
        avg_rating = restaurant["rating_avg"]

        # Generate ~50–80 reviews per restaurant
        num_reviews = random.randint(50, 80)

        for _ in range(num_reviews):
            # Weighted random rating around the average
            rating = int(np.clip(np.round(np.random.normal(avg_rating, 0.7)), 1, 5))

            # Pick review template based on rating
            if rating >= 4:
                template = random.choice(POSITIVE_REVIEWS)
                sentiment_label = "positive"
                sentiment_score = round(random.uniform(0.35, 1.0), 4)
            elif rating == 3:
                template = random.choice(NEUTRAL_REVIEWS)
                sentiment_label = "neutral"
                sentiment_score = round(random.uniform(-0.15, 0.35), 4)
            else:
                template = random.choice(NEGATIVE_REVIEWS)
                sentiment_label = "negative"
                sentiment_score = round(random.uniform(-1.0, -0.15), 4)

            dish = random.choice(dish_list)
            cuisine = random.choice(cuisine_list)
            review_text = template.format(dish=dish, cuisine=cuisine)

            # Simulate reviewer metadata
            reviewer_name = f"Reviewer_{review_id:04d}"
            visit_months = ["January", "February", "March", "April", "May", "June",
                            "July", "August", "September", "October", "November", "December"]
            visit_month = random.choice(visit_months)
            visit_year = random.choice([2022, 2023, 2024, 2025])

            records.append({
                "review_id": review_id,
                "restaurant_name": name,
                "address": restaurant["address"],
                "cuisine": ", ".join(cuisine_list),
                "price_range": restaurant["price_range"],
                "dish_mentioned": dish,
                "review_text": review_text,
                "rating": rating,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "reviewer": reviewer_name,
                "visit_month": visit_month,
                "visit_year": visit_year,
                "tags": ", ".join(restaurant["tags"]),
            })

            review_id += 1

    df = pd.DataFrame(records)
    return df, RESTAURANTS


def save_restaurant_metadata(restaurants, path):
    with open(path, "w") as f:
        json.dump(restaurants, f, indent=2)
    print(f"[✓] Restaurant metadata saved to {path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating restaurant reviews dataset for Majhitar, Sikkim...")
    df, restaurants = generate_reviews()

    raw_path = "data/majhitar_restaurant_reviews.csv"
    df.to_csv(raw_path, index=False)
    print(f"[✓] Dataset saved to {raw_path} ({len(df)} reviews, {df['restaurant_name'].nunique()} restaurants)")

    meta_path = "data/restaurant_metadata.json"
    save_restaurant_metadata(restaurants, meta_path)

    print(f"\n--- Dataset Summary ---")
    print(df["restaurant_name"].value_counts().to_string())
    print(f"\nSentiment Distribution:\n{df['sentiment_label'].value_counts().to_string()}")
    print(f"\nRating Distribution:\n{df['rating'].value_counts().sort_index().to_string()}")
