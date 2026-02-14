import json
import argparse
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from collections import Counter


def main(dim):
    results_dir = f"results/dim_{dim}"
    os.makedirs(results_dir, exist_ok=True)

    with open('dataset/updated_vocab_document_dict.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_words = len(data)
    print(f"Number of words: {num_words}")

    doc_text = {}

    for word, passages in data.items():
        for doc_idx, passage in passages:
            if doc_idx not in doc_text:
                doc_text[doc_idx] = passage

    # IMPORTANT: vocab order defines embedding rows
    vocabulary = sorted(data.keys())
    vocab_size = len(vocabulary)

    word_to_idx = {w: i for i, w in enumerate(vocabulary)}
    num_documents = max(doc_text.keys()) + 1

    row_indices = []
    col_indices = []
    values = []

    for doc_idx, text in doc_text.items():
        tokens = text.lower().split()
        token_counts = Counter(tokens)

        for word, count in token_counts.items():
            if word in word_to_idx:
                row_indices.append(word_to_idx[word])
                col_indices.append(doc_idx)
                values.append(count)

    term_doc_matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(vocab_size, num_documents)
    )

    print(f"Running SVD with dim={dim}")
    U, sigma, Vt = svds(term_doc_matrix, k=dim)

    # sort by descending singular values
    U = U[:, ::-1]
    sigma = sigma[::-1]
    Vt = Vt[::-1, :]

    Sigma_d = np.diag(sigma)
    embeddings = U @ Sigma_d

    print(f"Embeddings shape: {embeddings.shape}")

    # ---- save embeddings ----
    np.save(f"{results_dir}/embeddings_dim{dim}.npy", embeddings)

    # ---- save vocab (CRITICAL) ----
    with open(f"{results_dir}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings and vocab to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Embedding dimension (k for SVD)"
    )
    args = parser.parse_args()

    main(args.dim)
