Hate Speech Detection on Twitter Data

1. Introduction
This project focuses on the automatic detection of hate speech and offensive language in Twitter posts. The goal is to build machine learning models that can classify tweets into three categories:

Class 0: Hate speech
Class 1: Offensive language
Class 2: Neither

The dataset used is a publicly available labeled Twitter dataset containing 24,783 tweets. Each tweet was annotated by multiple workers, and the final class label was determined by majority vote.
2. Dataset Description
2.1 Source and Size

File: train.csv
Total samples: 24,783
Features:
count – number of annotators
hate_speech_count – number of annotators who labeled it as hate speech
offensive_language_count – number of annotators who labeled it as offensive
neither_count – number of annotators who labeled it as neither
class – final majority class (0, 1, or 2)
tweet – the original tweet text


2.2 Data Quality Checks

Duplicate rows: None found. After checking with df.duplicated(), the dataset remained at 24,783 rows.
Missing values: No missing values in any column (df.isnull().sum() returned all zeros).
Data types:
Numerical columns (count, hate_speech_count, etc.) are of type int64
tweet is of type object (string)


2.3 Class Distribution
From the descriptive statistics and count plot:

The dataset is imbalanced.
Class 1 (Offensive language) is the majority class.
Class 0 (Hate speech) is the minority class.
Class 2 (Neither) is also present but less frequent than Class 1.

3. Data Preprocessing
3.1 Text Cleaning
A custom cleaning function was applied to the tweet text:
Pythondef clean_text(text):
    text = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)
Steps performed:

Removal of leading/trailing non-alphabetic characters
Tokenization
Stop-word removal (using NLTK English stopwords)
Stemming using Porter Stemmer

A new column cleaned_tweet was created, and the original tweet column was dropped.
3.2 Sentiment Analysis
VADER (Valence Aware Dictionary and sEntiment Reasoner) was used to compute sentiment scores:

sentiment_score: Compound score ranging from -1 (most negative) to +1 (most positive)
sentiment: Categorical label
Positive (≥ 0.05)
Negative (≤ -0.05)
Neutral (between -0.05 and 0.05)


3.3 Feature Engineering
Several additional features were engineered:

Hate speech term indicators: Binary features indicating presence of specific hate-related terms
Linguistic markers:
Presence of offensive language
Excessive punctuation
Uppercase usage

Emoticon presence
N-gram features (converted to sparse matrix)

4. Feature Combination
The final feature matrix (combined_features) was constructed by horizontally stacking:

Padded token sequences (from Keras Tokenizer)
Sentiment scores
N-gram sparse features
Hate speech term features
Linguistic marker features
Emoticon features

Tokenization settings:

Vocabulary size: 5,000
Maximum sequence length: 100

5. Modeling
5.1 Classical Machine Learning Models
Two ensemble models were trained:
Random Forest Classifier

n_estimators = 100
random_state = 42
Test Accuracy: 0.8408 (84.08%)

Extra Trees Classifier

n_estimators = 100
random_state = 42
Test Accuracy: 0.8445 (84.45%)

Both models were evaluated using an 80/20 train-test split.
5.2 Neural Network Attempt
A hybrid architecture was attempted that combined:

Embedding layer + LSTM for sequential text processing
Concatenation with hand-crafted numerical features
Dense layers for final classification

However, the implementation contained structural issues (incorrect use of concatenate and mixing of sparse/dense inputs), and the model was not successfully trained in the notebook.

6. Results Summary

Model,Accuracy
Random Forest,84.08%
Extra Trees,84.45%




ModelAccuracyRandom Forest84.08%Extra Trees84.45%
Extra Trees slightly outperformed Random Forest on the test set.
7. Observations & Limitations
Strengths:

Clean dataset with no missing values or duplicates
Multiple complementary feature types (textual + linguistic + sentiment)
Strong baseline performance from tree-based ensembles

Limitations:

Class imbalance was not explicitly handled (no oversampling, undersampling, or class weighting)
Stemming was used instead of lemmatization in the main cleaning pipeline (although a lemmatized_tweet column appears later)
The neural network architecture was not correctly implemented
No detailed classification report (precision, recall, F1-score per class) was generated
No hyperparameter tuning was performed

8. Recommendations for Improvement

Address class imbalance using SMOTE, class weights, or stratified sampling.
Generate a full classification report and confusion matrix.
Perform hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
Replace stemming with lemmatization for better semantic preservation.
Properly implement a hybrid neural model (e.g., using Keras Functional API).
Experiment with modern transformers (BERT, RoBERTa) for stronger contextual understanding.
Add cross-validation for more reliable performance estimates.

9. Conclusion
This project successfully built a hate speech detection pipeline on a real-world Twitter dataset. Through careful preprocessing, feature engineering, and the use of ensemble models, an accuracy of approximately 84.5% was achieved using Extra Trees Classifier. While the results are promising, further improvements in handling class imbalance and exploring deeper language models would likely yield stronger performance, especially on the minority hate speech class.
