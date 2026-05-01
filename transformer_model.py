# ============================================
# INTERN FEEDBACK SENTIMENT ANALYSIS
# Model 2: Transformer (DistilBERT)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  INTERN FEEDBACK SENTIMENT ANALYSIS")
print("  Model: Transformer (DistilBERT)")
print("=" * 55)

# --------------------------------------------------
# STEP 1: Load Data
# --------------------------------------------------
df = pd.read_csv('data/intern_feedback.csv')
print(f"\n✅ Data Loaded: {len(df)} feedback entries")

# --------------------------------------------------
# STEP 2: Load Pre-trained Sentiment Pipeline
# --------------------------------------------------
print("\n⏳ Loading DistilBERT model (first run downloads ~250MB)...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True
)
print("✅ Model loaded!")

# --------------------------------------------------
# STEP 3: Map BERT labels → our labels
# --------------------------------------------------
def map_label(bert_label, score):
    if bert_label == 'POSITIVE' and score >= 0.80:
        return 'positive'
    elif bert_label == 'NEGATIVE' and score >= 0.80:
        return 'negative'
    else:
        return 'neutral'

# --------------------------------------------------
# STEP 4: Run Predictions on Full Dataset
# --------------------------------------------------
print("\n⏳ Running predictions...")
results  = sentiment_pipeline(df['feedback'].tolist())
df['predicted_label'] = [
    map_label(r['label'], r['score']) for r in results
]
df['confidence'] = [round(r['score'] * 100, 2) for r in results]
print("✅ Predictions complete!")

# --------------------------------------------------
# STEP 5: Show Results
# --------------------------------------------------
print("\n📋 Sample Predictions:")
print(df[['feedback', 'label', 'predicted_label',
          'confidence']].head(10).to_string(index=False))

print("\n📋 Classification Report:")
print(classification_report(df['label'], df['predicted_label'],
                             zero_division=0))

# --------------------------------------------------
# STEP 6: Confusion Matrix
# --------------------------------------------------
cm     = confusion_matrix(df['label'], df['predicted_label'],
         labels=['positive', 'neutral', 'negative'])
labels = ['Positive', 'Neutral', 'Negative']

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=labels, yticklabels=labels)
plt.title('Transformer (DistilBERT) — Confusion Matrix', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/transformer_confusion_matrix.png', dpi=150)
plt.show()
print("\n✅ Transformer confusion matrix saved!")

# --------------------------------------------------
# STEP 7: Confidence Distribution
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(df['confidence'], bins=10, color='#9b59b6',
         edgecolor='white', linewidth=1.2)
plt.title('DistilBERT — Prediction Confidence Distribution', fontsize=13)
plt.xlabel('Confidence (%)')
plt.ylabel('Number of Feedbacks')
plt.tight_layout()
plt.savefig('models/transformer_confidence.png', dpi=150)
plt.show()
print("✅ Confidence chart saved!")

# --------------------------------------------------
# STEP 8: Identify Improvement Areas
# --------------------------------------------------
print("\n" + "="*55)
print("  AREAS FOR IMPROVEMENT (Negative Feedbacks)")
print("="*55)

negative_df = df[df['predicted_label'] == 'negative']
print(f"\n⚠️  Total Negative Reviews: {len(negative_df)}\n")
for _, row in negative_df.iterrows():
    print(f"  ❌ {row['feedback']}")
    print(f"     Confidence: {row['confidence']}%\n")

# --------------------------------------------------
# STEP 9: Predict New Feedback
# --------------------------------------------------
print("="*55)
print("  LIVE PREDICTION ON NEW INTERN FEEDBACK")
print("="*55)

new_feedback = [
    "The mentorship program was absolutely wonderful",
    "Nobody helped me during the entire internship",
    "It was an average experience with some highs and lows"
]

new_results = sentiment_pipeline(new_feedback)
emoji_map   = {'positive': '😊', 'neutral': '😐', 'negative': '😟'}

for fb, res in zip(new_feedback, new_results):
    mapped = map_label(res['label'], res['score'])
    conf   = res['score'] * 100
    print(f"\n📝 Feedback : {fb}")
    print(f"   Sentiment: {emoji_map[mapped]} {mapped.upper()} ({conf:.1f}% confidence)")

print("\n✅ Transformer Analysis Complete!")