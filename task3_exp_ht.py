import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset
import re
import scipy.stats  # <--- FIXED: Added missing import
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# 1. Load dataset
dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

# 2. Define Helper for Word Shapes (Advanced Feature)
def get_word_shape(word):
    # Map characters to a simplified shape (e.g., Apple -> Xxxxx, CRF-2024 -> XXX-dddd)
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    shape = re.sub(r'[0-9]', 'd', shape)
    return shape

# 3. Define Advanced Feature Extractor
def word2features(sent_tokens, sent_pos, i):
    word = sent_tokens[i]
    postag = str(sent_pos[i])
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.shape': get_word_shape(word), # <--- ADDED: Word Shape
        'postag': postag,
        'is_first': i == 0,
        'is_last': i == len(sent_tokens) - 1,
        'is_capitalized': word[0].isupper() if word else False,
        'is_all_caps': word.isupper(),
        'is_all_lower': word.islower(),
        'has_hyphen': '-' in word,
        'is_numeric': word.isdigit(),
        'capitals_inside': any(c.isupper() for c in word[1:]) if len(word) > 1 else False,
    }
    
    # Logic for Abbreviations (U.S., O.J.)
    features['is_abbreviation'] = bool(re.match(r'^([A-Z]\.)+$', word))

    # Character N-Grams (Lengths 2 to 4 for broader coverage)
    for n in range(2, 5):
        if len(word) >= n:
            features[f'prefix-{n}'] = word[:n]
            features[f'suffix-{n}'] = word[-n:]

    # --- ADVANCED CONTEXTUAL WINDOWS (±2) ---
    
    # i-1 (Previous Word)
    if i > 0:
        word1 = sent_tokens[i-1]
        postag1 = str(sent_pos[i-1])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.shape': get_word_shape(word1),
            '-1:postag': postag1,
            'word_bigram_prev': f"{word1.lower()}_{word.lower()}", # <--- ADDED: Bigram
        })
    else:
        features['BOS'] = True

    # i-2 (Two words back)
    if i > 1:
        features.update({
            '-2:word.lower()': sent_tokens[i-2].lower(),
            '-2:postag': str(sent_pos[i-2]),
        })

    # i+1 (Next Word)
    if i < len(sent_tokens) - 1:
        word1 = sent_tokens[i+1]
        postag1 = str(sent_pos[i+1])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.shape': get_word_shape(word1),
            '+1:postag': postag1,
            'word_bigram_next': f"{word.lower()}_{word1.lower()}", # <--- ADDED: Bigram
        })
        # Conjunction Sandwich Logic (Bank [of] America)
        if word.lower() in ['of', 'and'] and i > 0:
            prev_w = sent_tokens[i-1]
            if prev_w[0].isupper() and word1[0].isupper():
                features['is_conjunction_sandwich'] = True
    else:
        features['EOS'] = True

    # i+2 (Two words ahead)
    if i < len(sent_tokens) - 2:
        features.update({
            '+2:word.lower()': sent_tokens[i+2].lower(),
            '+2:postag': str(sent_pos[i+2]),
        })

    return features

def sent2features(sentence_row):
    return [word2features(sentence_row['tokens'], sentence_row['pos_tags'], i) 
            for i in range(len(sentence_row['tokens']))]

def sent2labels(sent_tags):
    return [str(tag) for tag in sent_tags]

# 4. Process Data
print("Extracting features...")
X_train = [sent2features(row) for row in train_data]
y_train = [sent2labels(row['ner_tags']) for row in train_data]

X_test = [sent2features(row) for row in test_data]
y_test = [sent2labels(row['ner_tags']) for row in test_data]

# 5. Initialize Model and Search
# We do NOT call crf.fit() here yet. We let the search do it.
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# 6. Setup Scoring (Exclude '0' tag to focus on entities)
labels = list(set(tag for sent in y_train for tag in sent))
if '0' in labels: labels.remove('0')

f1_scorer = make_scorer(metrics.flat_f1_score, 
                        average='weighted', 
                        labels=labels)

# 7. Run Randomized Search
# Note: n_iter=20 is used to keep run time reasonable. Increase to 50 for better results.
print("Starting Hyperparameter Search (this may take a while)...")
rs = RandomizedSearchCV(
    crf,          # <--- FIXED: Passed the correct variable name
    params_space, 
    cv=3, 
    verbose=1, 
    n_jobs=-1, 
    n_iter=20, 
    scoring=f1_scorer
)

rs.fit(X_train, y_train)

# 8. Output Best Results
print('Best params:', rs.best_params_)
print('Best F1-score:', rs.best_score_)

# 9. Evaluate Best Model
crf_best = rs.best_estimator_
y_pred = crf_best.predict(X_test)

print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))

def print_top_features(crf_model, top_n=15):
    features = crf_model.state_features_
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{'Feature Name':<40} | {'Weight':<10}")
    print("-" * 55)
    for name, weight in sorted_features[:top_n]:
        print(f"{str(name):<40} | {weight:0.4f}")

print("Top features for the Best Model:")
print_top_features(crf_best) # <--- FIXED: Passing the best model, not the base one