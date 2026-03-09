import json
import numpy as np


# -------------------------------
# Build index from saved vocab
# -------------------------------
def buildIndex(words):
    word_idx = {word: i for i, word in enumerate(words)}
    idx_word = {i: word for i, word in enumerate(words)}
    return word_idx, idx_word


# -------------------------------
# Numerically safe cosine similarity
# -------------------------------
def cosine_similarity(vec, mat, eps=1e-8):
    vec_norm = np.linalg.norm(vec)
    if vec_norm < eps:
        return np.zeros(mat.shape[0])

    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norms[mat_norms < eps] = 1.0

    return (mat / mat_norms) @ (vec / vec_norm)


# -------------------------------
# Nearest neighbors by word
# -------------------------------
def nearest_neighbors_by_word(word, word_idx, idx_word, embeddings, k=10):
    word = word.lower()

    if word not in word_idx:
        print(f"'{word}' not in vocabulary")
        return

    idx = word_idx[word]
    vec = embeddings[idx]

    if np.linalg.norm(vec) == 0:
        print(f"\n'{word}' has zero embedding vector (cannot compute neighbors)")
        return

    sims = cosine_similarity(vec, embeddings)
    best = np.argsort(-sims)[1:k+1]  # skip itself

    print(f"\nNearest neighbors for '{word}':")
    for i in best:
        print(f"{idx_word[i]:20s}  {sims[i]:.4f}")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":

    dims = [50, 100, 200, 300]
    queries = ["queen", "10", "apple"]

    for dim in dims:
        print("\n" + "=" * 60)
        print(f"SVD Nearest Neighbors (dim={dim})")
        print("=" * 60)

        # ---- load embeddings ----
        emb_path = f"results/dim_{dim}/embeddings_dim{dim}.npy"
        embeddings = np.load(emb_path)
        print(f"Loaded embeddings with shape: {embeddings.shape}")

        # ---- load vocab ----
        vocab_path = f"results/dim_{dim}/vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        word_idx, idx_word = buildIndex(vocab)

        # ---- sanity check ----
        assert embeddings.shape[0] == len(vocab), \
            f"Embedding/vocab mismatch for dim={dim}"

        # ---- queries ----
        for q in queries:
            nearest_neighbors_by_word(
                word=q,
                word_idx=word_idx,
                idx_word=idx_word,
                embeddings=embeddings,
                k=10
            )

