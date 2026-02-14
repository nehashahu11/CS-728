import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset
import re
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

def get_word_shape(word):
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
        'word.shape': get_word_shape(word),
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
    
    features['is_abbreviation'] = bool(re.match(r'^([A-Z]\.)+$', word))

    for n in range(2, 5):
        if len(word) >= n:
            features[f'prefix-{n}'] = word[:n]
            features[f'suffix-{n}'] = word[-n:]
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

    if i > 1:
        features.update({
            '-2:word.lower()': sent_tokens[i-2].lower(),
            '-2:postag': str(sent_pos[i-2]),
        })

    if i < len(sent_tokens) - 1:
        word1 = sent_tokens[i+1]
        postag1 = str(sent_pos[i+1])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.shape': get_word_shape(word1),
            '+1:postag': postag1,
            'word_bigram_next': f"{word.lower()}_{word1.lower()}",
        })
        if word.lower() in ['of', 'and'] and i > 0:
            prev_w = sent_tokens[i-1]
            if prev_w[0].isupper() and word1[0].isupper():
                features['is_conjunction_sandwich'] = True
    else:
        features['EOS'] = True

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

print("Extracting features...")
X_train = [sent2features(row) for row in train_data]
y_train = [sent2labels(row['ner_tags']) for row in train_data]

X_test = [sent2features(row) for row in test_data]
y_test = [sent2labels(row['ner_tags']) for row in test_data]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

labels = list(set(tag for sent in y_train for tag in sent))
if '0' in labels: labels.remove('0')

f1_scorer = make_scorer(metrics.flat_f1_score, 
                        average='weighted', 
                        labels=labels)


rs = RandomizedSearchCV(
    crf,
    params_space, 
    cv=3, 
    verbose=1, 
    n_jobs=-1, 
    n_iter=20, 
    scoring=f1_scorer
)

rs.fit(X_train, y_train)

print('Best params:', rs.best_params_)
print('Best F1-score:', rs.best_score_)

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
print_top_features(crf_best, top_n=20)
accuracy = metrics.flat_accuracy_score(y_test, y_pred)
print(f"\nToken-level Accuracy: {accuracy:.4f}")

#accuracy for entity tokens

y_test_flat = [tag for sent in y_test for tag in sent]
y_pred_flat = [tag for sent in y_pred for tag in sent]


correct_entities = 0
total_entities = 0
for true, pred in zip(y_test_flat, y_pred_flat):
    if true != '0':  
        total_entities += 1
        if true == pred:
            correct_entities += 1

entity_accuracy = correct_entities / total_entities if total_entities > 0 else 0
print(f"Entity-only Accuracy: {entity_accuracy:.4f}")