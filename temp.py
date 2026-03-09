import json
import numpy as np


# -------------------------------
# Build index (must match training)
# -------------------------------
def buildIndex(words):
    '''
    This function takes a list of words and returns two dictionary.
    One word to idx and another idx to word
    '''
    word_idx = {word: i for i, word in enumerate(words)}
    idx_word = {i: word for i, word in enumerate(words)}
    return word_idx, idx_word


# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(vec, mat):
    vec_norm = vec / np.linalg.norm(vec)
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    return mat_norm @ vec_norm


# -------------------------------
# Nearest neighbors by word
# -------------------------------
def nearest_neighbors_by_word(word, word_idx, idx_word, embeddings, k=10):
    if word not in word_idx:
        print(f"'{word}' not in vocabulary")
        return

    idx = word_idx[word]
    vec = embeddings[idx]

    sims = cosine_similarity(vec, embeddings)
    best = np.argsort(-sims)[1:k+1]  # skip itself

    print(f"\nNearest neighbors for '{word}':")
    for i in best:
        print(f"{idx_word[i]:20s}  {sims[i]:.4f}")


# -------------------------------
# Nearest neighbors by index
# -------------------------------
def nearest_neighbors_by_index(index, idx_word, embeddings, k=10):
    vec = embeddings[index]

    sims = cosine_similarity(vec, embeddings)
    best = np.argsort(-sims)[1:k+1]

    print(f"\nNearest neighbors for index {index} ('{idx_word[index]}'):")
    for i in best:
        print(f"{idx_word[i]:20s}  {sims[i]:.4f}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    # load vocab (same file used in training)
    with open("./dataset/updated_vocab_document_dict.json", "r") as f:
        data = json.load(f)

    words = list(data.keys())
    word_idx, idx_word = buildIndex(words)

    # load 50-dim embeddings (W1 + W2)
    embeddings = np.load("results/embeddings_dim50.npy")

    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # ---- queries ----
    nearest_neighbors_by_word(
        word="Queen",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="10",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="Apple",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
   )
    embeddings = np.load("results/embeddings_dim100.npy")

    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # ---- queries ----
    nearest_neighbors_by_word(
        word="Queen",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="10",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="Apple",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
   )
    embeddings = np.load("results/embeddings_dim200.npy")

    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # ---- queries ----
    nearest_neighbors_by_word(
        word="Queen",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="10",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="Apple",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
   )
    embeddings = np.load("results/embeddings_dim300.npy")

    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # ---- queries ----
    nearest_neighbors_by_word(
        word="Queen",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="10",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
    )
    nearest_neighbors_by_word(
        word="Apple",
        word_idx=word_idx,
        idx_word=idx_word,
        embeddings=embeddings,
        k=10
   )
