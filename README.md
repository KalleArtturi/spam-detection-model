# 📧 Spam Detection Model

A machine learning pipeline that classifies text messages as **spam** or **ham** (legitimate), built with pandas and scikit-learn. The project is split into stage-by-stage Jupyter notebooks, each saving its output so the next stage can pick up where the last one left off.

This is a learning project, built while following Hack The Box Academy training and extended into a multi-stage pipeline.

## Project Structure

```
├── data/                       datasets for each pipeline stage (not tracked by git)
├── models/                     trained model pipeline (not tracked by git)
├── importdata.ipynb            1. download and load the raw dataset
├── preprocessing.ipynb         2. clean and normalize the message text
├── feature_extraction.ipynb    3. explore text features and vectorizer settings
├── training.ipynb              4. tune, train, and evaluate the classifier
├── model test.ipynb            5. classify new, unseen messages
└── README.md
```

## Pipeline Overview

**1. Import data** — Loads the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset into a pandas DataFrame with a `label` and `message` column.

**2. Preprocessing** — Cleans the raw text so the model focuses on meaningful content: lowercasing, removing punctuation and numbers, tokenizing, removing stop words, and stemming. The cleaned data is saved as Parquet to preserve data types between stages.

**3. Feature extraction** — Explores how messages are converted into word-count vectors with `CountVectorizer`, including unigrams and bigrams. This stage is used to understand the vocabulary and choose sensible vectorizer settings, which are then tuned in training.

**4. Training** — Builds a scikit-learn `Pipeline` combining `CountVectorizer` and a Multinomial Naive Bayes classifier. The data is split 80/20 into training and test sets (stratified to keep the spam ratio equal), and `GridSearchCV` with 5-fold cross-validation tunes both the vectorizer and the classifier, optimizing for F1-score. Because vectorizing happens inside the pipeline, the test set never influences the vocabulary, which prevents data leakage. The best model is evaluated on the held-out test set and saved with joblib.

**5. Model test** — Loads the saved pipeline and classifies new messages after applying the same preprocessing as the training data.

## Results

Evaluated on a held-out test set of 1,034 messages (903 ham, 131 spam). Scores below are for the spam class:

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.98  |
| Precision | 0.91  |
| Recall    | 0.89  |
| F1-score  | 0.90  |

| | Predicted ham | Predicted spam |
|---|---|---|
| **Actual ham** | 892 | 11 |
| **Actual spam** | 14 | 117 |

The model catches 89% of spam while flagging only 11 of 903 legitimate messages. Since about 87% of messages are ham, the spam-class scores are a more meaningful measure than overall accuracy.

## Getting Started

### Requirements

- Python 3.x with conda
- pandas, scikit-learn, pyarrow, nltk, joblib, matplotlib

### Setup

```bash
conda create -n spam python
conda activate spam
conda install -c conda-forge pandas scikit-learn pyarrow nltk joblib matplotlib jupyterlab
```

### Running the pipeline

Run the notebooks **in order**, since each one depends on the files created by the previous stage:

1. `importdata.ipynb`
2. `preprocessing.ipynb`
3. `feature_extraction.ipynb` (optional, exploration only)
4. `training.ipynb`
5. `model test.ipynb`

The `data/` and `models/` folders are excluded from git, so running the full pipeline regenerates everything from scratch.

> **Note:** Saved `.joblib` files are tied to the scikit-learn version used to create them. Use the same environment for every stage.

## Future Improvements

- Analyze misclassified messages to find patterns the model misses
- Try TF-IDF weighting instead of raw word counts
- Compare against other classifiers (Logistic Regression, SVM)
- Adjust the decision threshold to reduce false positives
- Wrap the model in a simple API or CLI for real-time predictions