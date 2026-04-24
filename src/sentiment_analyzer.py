"""
Self-contained Sentiment Analysis — no external NLP packages needed.
Uses sklearn + pure Python lexicons.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

VADER_LEXICON = {
    "amazing": 3.1, "excellent": 3.2, "outstanding": 3.4, "fantastic": 3.3,
    "wonderful": 3.0, "superb": 3.3, "brilliant": 3.1, "perfect": 3.5,
    "extraordinary": 3.4, "phenomenal": 3.5, "exceptional": 3.3,
    "delicious": 3.4, "mouthwatering": 3.2, "scrumptious": 3.1, "heavenly": 3.3,
    "loved": 2.9, "love": 2.9, "best": 2.8, "awesome": 3.1,
    "good": 1.9, "great": 2.5, "nice": 1.8, "tasty": 2.4, "fresh": 2.0,
    "recommend": 2.2, "enjoyable": 2.1, "satisfied": 2.0, "happy": 2.1,
    "pleasant": 2.0, "yummy": 2.5, "flavorful": 2.3, "tender": 2.1,
    "crispy": 1.8, "juicy": 2.0, "rich": 1.7, "authentic": 2.0,
    "friendly": 1.9, "welcoming": 1.8, "quick": 1.5, "clean": 1.6,
    "reasonable": 1.5, "worth": 1.7, "value": 1.5, "hot": 1.4,
    "cozy": 1.7, "comfortable": 1.6, "charming": 1.8, "beautiful": 1.9,
    "okay": 0.6, "decent": 0.9, "alright": 0.7, "fine": 0.6, "acceptable": 0.5,
    "adequate": 0.4, "passable": 0.5, "edible": 0.3,
    "average": -0.4, "ordinary": -0.5, "mediocre": -1.3, "bland": -1.8,
    "slow": -1.2, "cold": -1.6, "overpriced": -2.0, "expensive": -1.2,
    "bad": -2.0, "terrible": -3.2, "horrible": -3.3, "awful": -3.1,
    "disgusting": -3.5, "worst": -3.4, "disappointing": -2.5, "pathetic": -2.8,
    "tasteless": -2.3, "stale": -2.2, "undercooked": -2.4, "overcooked": -2.1,
    "dirty": -2.8, "rude": -2.6, "unfriendly": -2.2, "waste": -2.5,
    "soggy": -1.9, "oily": -1.7, "salty": -1.5, "bitter": -1.6,
    "unhygienic": -3.0, "smell": -2.0, "stench": -3.0, "regret": -2.5,
}
NEGATION_WORDS = {"not","never","no","neither","nor","without","nobody","nothing",
                  "nowhere","hardly","barely","dont","doesnt","didnt","wont","cant",
                  "couldnt","shouldnt","wouldnt","isnt","arent","wasnt","werent"}
BOOSTER_MAP = {"very":0.293,"extremely":0.293,"incredibly":0.293,"absolutely":0.293,
               "really":0.293,"so":0.293,"totally":0.293,"highly":0.293,"super":0.293,
               "quite":0.1,"rather":0.1,"somewhat":-0.2,"slightly":-0.2,"little":-0.2}
SUBJECTIVITY_LEXICON = {k:(v/4.0,0.8) for k,v in VADER_LEXICON.items() if abs(v)>0.5}


class VADERSentimentAnalyzer:
    name = "VADER"
    def _tok(self, t): return re.findall(r"[a-z']+", t.lower())
    def predict_single(self, text):
        tokens = self._tok(str(text))
        scores = []
        for i, token in enumerate(tokens):
            if token in VADER_LEXICON and abs(VADER_LEXICON[token]) > 0.5:
                score = VADER_LEXICON[token]
                boost = sum(BOOSTER_MAP.get(tokens[j],0) for j in range(max(0,i-3),i))
                neg = any(tokens[j] in NEGATION_WORDS for j in range(max(0,i-3),i))
                if boost: score += boost * np.sign(score)
                if neg: score *= -0.74
                scores.append(score)
        raw = sum(scores) if scores else 0.0
        compound = raw / np.sqrt(raw**2 + 15.0) if raw != 0 else 0.0
        pos_s = sum(s for s in scores if s > 0) if scores else 0
        neg_s = abs(sum(s for s in scores if s < 0)) if scores else 0
        total = pos_s + neg_s + 0.001
        pos = round(pos_s/total, 3); neg_r = round(neg_s/total, 3)
        label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        return {"label":label,"compound":round(float(compound),4),"pos":pos,"neu":round(max(0.0,1-pos-neg_r),3),"neg":neg_r}
    def predict_batch(self, texts): return pd.DataFrame([self.predict_single(t) for t in texts])
    def evaluate(self, texts, true_labels):
        preds = [self.predict_single(t)["label"] for t in texts]
        return {"model":self.name,"accuracy":round(accuracy_score(true_labels,preds),4),
                "f1_weighted":round(f1_score(true_labels,preds,average="weighted",zero_division=0),4),
                "report":classification_report(true_labels,preds,zero_division=0),
                "confusion_matrix":confusion_matrix(true_labels,preds,labels=["positive","neutral","negative"]),
                "predictions":preds}


class TextBlobSentimentAnalyzer:
    name = "TextBlob"
    def _tok(self, t): return re.findall(r"[a-z']+", t.lower())
    def predict_single(self, text):
        tokens = self._tok(str(text))
        pols, subjs = [], []
        for i, tok in enumerate(tokens):
            if tok in SUBJECTIVITY_LEXICON:
                pol, subj = SUBJECTIVITY_LEXICON[tok]
                neg = any(tokens[j] in NEGATION_WORDS for j in range(max(0,i-3),i))
                pols.append(pol * (-0.5 if neg else 1))
                subjs.append(subj)
        polarity = float(np.mean(pols)) if pols else 0.0
        subjectivity = float(np.mean(subjs)) if subjs else 0.0
        polarity = max(-1.0, min(1.0, polarity))
        label = "positive" if polarity > 0.05 else ("negative" if polarity < -0.05 else "neutral")
        return {"label":label,"polarity":round(polarity,4),"subjectivity":round(subjectivity,4)}
    def predict_batch(self, texts): return pd.DataFrame([self.predict_single(t) for t in texts])
    def evaluate(self, texts, true_labels):
        preds = [self.predict_single(t)["label"] for t in texts]
        return {"model":self.name,"accuracy":round(accuracy_score(true_labels,preds),4),
                "f1_weighted":round(f1_score(true_labels,preds,average="weighted",zero_division=0),4),
                "report":classification_report(true_labels,preds,zero_division=0),
                "confusion_matrix":confusion_matrix(true_labels,preds,labels=["positive","neutral","negative"]),
                "predictions":preds}


class MLSentimentClassifier:
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type
        self.pipeline = None
        self.is_trained = False
        self.name = {"logistic_regression":"Logistic Regression","naive_bayes":"Naive Bayes","svm":"Linear SVM"}.get(model_type,model_type)
    def _build(self):
        tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,max_df=0.95,sublinear_tf=True)
        clf = (LogisticRegression(max_iter=500,C=1.0,solver="lbfgs",random_state=42)
               if self.model_type=="logistic_regression" else
               (MultinomialNB(alpha=1.0) if self.model_type=="naive_bayes" else
                LinearSVC(max_iter=1000,C=1.0,random_state=42)))
        return Pipeline([("tfidf",tfidf),("clf",clf)])
    def train(self, texts, labels):
        print(f"  [→] Training {self.name}...")
        self.pipeline = self._build()
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        print(f"  [✓] Done ({len(texts)} samples)")
    def predict_single(self, text):
        label = self.pipeline.predict([text])[0]
        try: conf = float(max(self.pipeline.predict_proba([text])[0]))
        except: conf = 1.0
        return {"label":label,"confidence":round(conf,4)}
    def predict_batch(self, texts): return list(self.pipeline.predict(texts))
    def evaluate(self, texts, true_labels):
        preds = self.predict_batch(texts)
        return {"model":self.name,"accuracy":round(accuracy_score(true_labels,preds),4),
                "f1_weighted":round(f1_score(true_labels,preds,average="weighted",zero_division=0),4),
                "report":classification_report(true_labels,preds,zero_division=0),
                "confusion_matrix":confusion_matrix(true_labels,preds,labels=["positive","neutral","negative"]),
                "predictions":preds}
    def cross_validate(self, texts, labels, cv=5):
        scores = cross_val_score(self._build(),texts,labels,cv=cv,scoring="f1_weighted")
        return {"model":self.name,"cv_scores":scores.tolist(),"cv_mean":round(scores.mean(),4),"cv_std":round(scores.std(),4)}
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self,f)
        print(f"  [✓] Saved: {path}")
    @classmethod
    def load(cls, path):
        with open(path,"rb") as f: return pickle.load(f)


class EnsembleSentimentAnalyzer:
    name = "Ensemble (VADER + TextBlob + ML)"
    def __init__(self, ml_model=None):
        self.vader = VADERSentimentAnalyzer()
        self.textblob = TextBlobSentimentAnalyzer()
        self.ml_model = ml_model
    def predict_single(self, text):
        v = self.vader.predict_single(text)
        tb = self.textblob.predict_single(text)
        votes = [v["label"], tb["label"]]
        if self.ml_model and self.ml_model.is_trained:
            votes.append(self.ml_model.predict_single(text)["label"])
        counts = Counter(votes)
        final = counts.most_common(1)[0][0]
        return {"label":final,"confidence":round(counts[final]/len(votes),4),
                "vader_label":v["label"],"textblob_label":tb["label"],
                "vader_compound":v["compound"],"textblob_polarity":tb["polarity"]}
    def predict_batch(self, texts): return [self.predict_single(t) for t in texts]
    def evaluate(self, texts, true_labels):
        preds = [self.predict_single(t)["label"] for t in texts]
        return {"model":self.name,"accuracy":round(accuracy_score(true_labels,preds),4),
                "f1_weighted":round(f1_score(true_labels,preds,average="weighted",zero_division=0),4),
                "report":classification_report(true_labels,preds,zero_division=0),
                "confusion_matrix":confusion_matrix(true_labels,preds,labels=["positive","neutral","negative"]),
                "predictions":preds}


def train_all_models(df, text_col="cleaned_text", label_col="sentiment_label"):
    print("\n" + "="*60)
    print("  TRAINING ALL SENTIMENT MODELS")
    print("="*60)
    texts = df[text_col].tolist(); labels = df[label_col].tolist()
    X_train,X_test,y_train,y_test = train_test_split(texts,labels,test_size=0.2,random_state=42,stratify=labels)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    results = {}; trained_models = {}
    for cls in [VADERSentimentAnalyzer, TextBlobSentimentAnalyzer]:
        a = cls(); r = a.evaluate(X_test, y_test)
        results[a.name] = r; trained_models[a.name] = a
        print(f"  [{a.name}] Acc: {r['accuracy']} | F1: {r['f1_weighted']}")
    best_ml = None; best_f1 = 0.0
    for mt in ["logistic_regression","naive_bayes","svm"]:
        clf = MLSentimentClassifier(mt); clf.train(X_train, y_train)
        r = clf.evaluate(X_test, y_test); cv = clf.cross_validate(texts, labels, cv=5)
        r["cv_mean"] = cv["cv_mean"]; r["cv_std"] = cv["cv_std"]
        results[clf.name] = r; trained_models[clf.name] = clf
        print(f"  [{clf.name}] Acc: {r['accuracy']} | F1: {r['f1_weighted']} | CV: {cv['cv_mean']}±{cv['cv_std']}")
        if r["f1_weighted"] > best_f1: best_f1 = r["f1_weighted"]; best_ml = clf
    ens = EnsembleSentimentAnalyzer(ml_model=best_ml); er = ens.evaluate(X_test, y_test)
    results[ens.name] = er; trained_models[ens.name] = ens
    print(f"  [{ens.name}] Acc: {er['accuracy']} | F1: {er['f1_weighted']}")
    os.makedirs("models", exist_ok=True)
    if best_ml: best_ml.save("models/best_ml_model.pkl")
    print("\n  TRAINING COMPLETE")
    return results, trained_models, (X_train, X_test, y_train, y_test)
