# ============================================
# INTERN FEEDBACK SENTIMENT ANALYSIS
# Model 1: Logistic Regression
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  INTERN FEEDBACK SENTIMENT ANALYSIS")
print("  Model: Logistic Regression")
print("=" * 55)

# --------------------------------------------------
# STEP 1: Load Data
# --------------------------------------------------
df = pd.read_csv('data/intern_feedback.csv')
print(f"\n✅ Data Loaded: {len(df)} feedback entries")
print("\n📊 Label Distribution:")
print(df['label'].value_counts())

# --------------------------------------------------
# STEP 2: Preprocessing
# --------------------------------------------------
df['feedback'] = df['feedback'].str.lower().str.strip()

# --------------------------------------------------
# STEP 3: Split Data
# --------------------------------------------------
X = df['feedback']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n✅ Train size: {len(X_train)} | Test size: {len(X_test)}")

# --------------------------------------------------
# STEP 4: TF-IDF Vectorization
# --------------------------------------------------
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print("✅ TF-IDF Vectorization complete")

# --------------------------------------------------
# STEP 5: Train Model
# --------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)
print("✅ Model training complete")

# --------------------------------------------------
# STEP 6: Evaluate
# --------------------------------------------------
y_pred    = model.predict(X_test_vec)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  ACCURACY: {accuracy * 100:.2f}%")
print(f"{'='*40}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# STEP 7: Confusion Matrix Plot
# --------------------------------------------------
cm     = confusion_matrix(y_test, y_pred,
         labels=['positive', 'neutral', 'negative'])
labels = ['Positive', 'Neutral', 'Negative']

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Logistic Regression — Confusion Matrix', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/lr_confusion_matrix.png', dpi=150)
plt.show()
print("\n✅ Confusion matrix saved!")

# --------------------------------------------------
# STEP 8: Sentiment Distribution Plot
# --------------------------------------------------
colors = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
counts = df['label'].value_counts()

plt.figure(figsize=(7, 4))
bars = plt.bar(counts.index, counts.values,
               color=[colors[l] for l in counts.index], edgecolor='white',
               linewidth=1.2)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1, str(val),
             ha='center', fontweight='bold')
plt.title('Intern Feedback — Sentiment Distribution', fontsize=14)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('models/sentiment_distribution.png', dpi=150)
plt.show()
print("✅ Distribution chart saved!")

# --------------------------------------------------
# STEP 9: Predict on New Feedback
# --------------------------------------------------
print("\n" + "="*55)
print("  LIVE PREDICTION ON NEW INTERN FEEDBACK")
print("="*55)

new_feedback = [
    "The mentorship was incredibly helpful and supportive",
    "The tasks were boring and I learned nothing new",
    "The experience was okay, not great but not bad either"
]

new_vec   = vectorizer.transform(new_feedback)
new_preds = model.predict(new_vec)
new_probs = model.predict_proba(new_vec)
classes   = model.classes_

emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😟'}

for feedback, pred, prob in zip(new_feedback, new_preds, new_probs):
    confidence = max(prob) * 100
    print(f"\n📝 Feedback : {feedback}")
    print(f"   Sentiment: {emoji_map[pred]} {pred.upper()} ({confidence:.1f}% confidence)")

print("\n✅ Logistic Regression Analysis Complete!")