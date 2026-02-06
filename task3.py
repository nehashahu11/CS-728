import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset
dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
# dataset.save_to_disk("conll-2003")
train_data = dataset['train']
test_data = dataset['test']
val_data = dataset['validation']

def word2features(sent_tokens, sent_pos, i):
    word = sent_tokens[i]
    postag = str(sent_pos[i]) # Use the POS tag ID as a feature
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag, # Current POS tag
    }
    
    # Features for the PREVIOUS word
    if i > 0:
        word1 = sent_tokens[i-1]
        postag1 = str(sent_pos[i-1])
        features.update({
            '-1:word.islower()': word1.islower(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1, # Previous POS tag
        })
    else:
        features['BOS'] = True

    # Features for the NEXT word
    if i < len(sent_tokens) - 1:
        word1 = sent_tokens[i+1]
        postag1 = str(sent_pos[i+1])
        features.update({
            '+1:word.islower()': word1.islower(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1, # Next POS tag
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sentence_row):
    # Pass both tokens and pos_tags from the dataset row
    return [word2features(sentence_row['tokens'], sentence_row['pos_tags'], i) 
            for i in range(len(sentence_row['tokens']))]


def sent2labels(sent_tags):
    return [str(tag) for tag in sent_tags]


# Extract features and labels for Train and Test
X_train = [sent2features(row) for row in train_data]
y_train = [sent2labels(row['ner_tags']) for row in train_data]

X_test = [sent2features(row) for row in dataset['test']]
y_test = [sent2labels(row['ner_tags']) for row in dataset['test']]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,         # L1 regularization (encourages sparsity)
    c2=0.1,         # L2 regularization (prevents overfitting)
    max_iterations=100,
    all_possible_transitions=True # Allows model to learn transitions not in data
)

crf.fit(X_train, y_train)


# Predict tags for the test set
y_pred = crf.predict(X_test)

# Get the list of all labels except '0' (Outside) for cleaner reporting
labels = list(crf.classes_)
labels.remove('0') # Assuming '0' is the ID for 'O' in your dataset

# Print detailed evaluation
print(metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
))


def print_top_features(crf, top_n=10):
    # Get the state features and their weights
    features = crf.state_features_
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{'Feature Name':<30} | {'Weight':<10}")
    print("-" * 45)
    for name, weight in sorted_features[:top_n]:
        print(f"{str(name):<30} | {weight:0.4f}")

print_top_features(crf)