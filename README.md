# Yelp Review Sentiment Classification

A binary sentiment classifier for Yelp reviews. Reviews with 4-5 stars are labeled "good" and reviews with 1-3 stars are labeled "bad," then classified using a bag-of-words representation and logistic regression.

## Overview

- **Task:** Binary sentiment classification (good vs. bad review)
- **Data:** [Yelp review dataset](https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%203%20-%20NLP/yelp_final.csv) (business ID, review text, star rating, user metadata)
- **Features:** Bag-of-words vectors (800 features) built from spaCy-lemmatized, stopword- and punctuation-filtered tokens
- **Model:** Logistic Regression (scikit-learn)
- **Result:** 71.5% test accuracy, vs. a 68.5% majority-class baseline (see [Results & Limitations](#results--limitations))

## Project structure

```
.
├── notebooks/
│   └── yelp_review_sentiment_classification.ipynb   # end-to-end analysis and model
├── requirements.txt
└── README.md
```

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

The notebook also downloads a couple of NLTK resources on first run:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 2. Run the notebook

Open [`notebooks/yelp_review_sentiment_classification.ipynb`](notebooks/yelp_review_sentiment_classification.ipynb) in Jupyter or Google Colab and run the cells in order. The first code cell downloads the dataset (`yelp_final.csv`) automatically.

## Approach

1. **Data exploration** — load the raw Yelp dataset, inspect the review text and star ratings, and visualize word frequency per rating with word clouds.
2. **Label preparation** — reduce the 1-5 star rating to a binary label: 4-5 stars → good, 1-3 stars → bad.
3. **Text preprocessing** — tokenize and lemmatize review text with spaCy, filtering out stop words and punctuation.
4. **Feature extraction** — vectorize the preprocessed text into a bag-of-words matrix (top 800 tokens) using `CountVectorizer`.
5. **Modeling** — split the data into train/test sets and fit a logistic regression classifier.
6. **Evaluation** — score the model against a majority-class baseline, and break down performance with a confusion matrix and per-class precision/recall.

## Results & Limitations

- **The model beats the naive baseline, but not by a wide margin.** Logistic regression scores 71.5% vs. 68.5% for always predicting "good." A meaningful chunk of the raw accuracy number comes from class imbalance rather than the model learning to distinguish sentiment.
- **The model struggles most on "bad" reviews.** Recall on the minority class is only 0.35, meaning most negative reviews get misclassified as positive — the failure mode that would matter most in a real moderation or alerting use case.
- **The dataset is small and the feature space is narrow.** ~3,000 reviews and an 800-token bag-of-words vocabulary is enough for a proof of concept but limits how much nuance the model can capture.
- **Possible next steps:** address class imbalance (`class_weight='balanced'`, resampling, or threshold tuning), try TF-IDF weighting and a non-linear model, use k-fold cross-validation instead of a single split, or keep the original 1-5 star scale as an ordinal/multiclass target instead of collapsing to binary.

## License

No license specified. Add one if you plan to share or reuse this project.
