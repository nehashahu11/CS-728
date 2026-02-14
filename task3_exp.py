import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset
import re
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# Load dataset
dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

import re

def get_word_shape(word):
    # Map characters to a simplified shape (e.g., Apple -> Xxxxx, CRF-2024 -> XXX-dddd)
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    shape = re.sub(r'[0-9]', 'd', shape)
    return shape

def word2features(sent_tokens, sent_pos, i):
    word = sent_tokens[i]
    postag = str(sent_pos[i])
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.shape': get_word_shape(word), # Word Shape Grouping
        'postag': postag,
        'is_all_caps': word.isupper(),
        'is_istitle': word.istitle(),
        'is_digit': word.isdigit(),
        'capitals_inside': any(c.isupper() for c in word[1:]) if len(word) > 1 else False,
    }

    # Character-Level N-Grams (Lengths 2 to 5)
    for n in range(2, 6):
        if len(word) >= n:
            features[f'prefix-{n}'] = word[:n]
            features[f'suffix-{n}'] = word[-n:]

    # --- ADVANCED CONTEXTUAL FEATURES & WINDOW EXPANSION ---
    
    # i-1 (Previous Word)
    if i > 0:
        word1 = sent_tokens[i-1]
        postag1 = str(sent_pos[i-1])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.shape': get_word_shape(word1),
            '-1:postag': postag1,
            # Bigram Feature: word[i-1] + word[i]
            'word_bigram_prev': f"{word1.lower()}_{word.lower()}",
        })
    else:
        features['BOS'] = True

    # i-2 (Two words back)
    if i > 1:
        word2 = sent_tokens[i-2]
        postag2 = str(sent_pos[i-2])
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:postag': postag2,
        })

    # i+1 (Next Word)
    if i < len(sent_tokens) - 1:
        word1 = sent_tokens[i+1]
        postag1 = str(sent_pos[i+1])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.shape': get_word_shape(word1),
            '+1:postag': postag1,
            # Bigram Feature: word[i] + word[i+1]
            'word_bigram_next': f"{word.lower()}_{word1.lower()}",
        })
        # Special logic for "Bank of America" style organizations
        if word.lower() in ['of', 'and'] and i > 0:
            prev_w = sent_tokens[i-1]
            if prev_w[0].isupper() and word1[0].isupper():
                features['is_conjunction_sandwich'] = True
    else:
        features['EOS'] = True

    # i+2 (Two words ahead)
    if i < len(sent_tokens) - 2:
        word2 = sent_tokens[i+2]
        postag2 = str(sent_pos[i+2])
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:postag': postag2,
        })

    return features

def sent2features(sentence_row):
    return [word2features(sentence_row['tokens'], sentence_row['pos_tags'], i) 
            for i in range(len(sentence_row['tokens']))]

def sent2labels(sent_tags):
    return [str(tag) for tag in sent_tags]

# Extracting features
X_train = [sent2features(row) for row in train_data]
y_train = [sent2labels(row['ner_tags']) for row in train_data]

X_test = [sent2features(row) for row in test_data]
y_test = [sent2labels(row['ner_tags']) for row in test_data]

# Training with L-BFGS and L1/L2 Regularization
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,         
    c2=0.1,         
    max_iterations=100,
    all_possible_transitions=True 
)

crf.fit(X_train, y_train)

# Evaluation
y_pred = crf.predict(X_test)
labels = list(crf.classes_)
if '0' in labels: labels.remove('0')

print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))

def print_top_features(crf, top_n=15):
    features = crf.state_features_
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{'Feature Name':<40} | {'Weight':<10}")
    print("-" * 55)
    for name, weight in sorted_features[:top_n]:
        print(f"{str(name):<40} | {weight:0.4f}")

print_top_features(crf)