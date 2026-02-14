import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("="*60)
print("TASK 5: TF-IDF SVD IMPLEMENTATION")
print("="*60)

# ========================================
# LOAD DATA & BUILD TERM-DOC MATRIX
# ========================================

print("\n[1/6] Loading data...")
with open('dataset/updated_vocab_document_dict.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

num_words = len(data)
print(f"Number of words: {num_words}")

vocabulary = list(data.keys())
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

# Find number of documents
max_doc_idx = 0
for word, passages in data.items():
    for doc_idx, passage in passages:
        if doc_idx > max_doc_idx:
            max_doc_idx = doc_idx

num_documents = max_doc_idx + 1
vocab_size = len(vocabulary)

print(f"Vocabulary size: {vocab_size}")
print(f"Number of documents: {num_documents}")

# ========================================
# BUILD TERM-DOCUMENT MATRIX
# ========================================

print("\n[2/6] Building term-document matrix...")

row_indices = [] 
col_indices = [] 
values = []

for word in vocabulary:
    word_idx = word_to_idx[word]
    passages = data[word]
    
    for doc_idx, passage in passages:
        tokens = passage.lower().split()
        count = tokens.count(word.lower())
        
        row_indices.append(word_idx)
        col_indices.append(doc_idx)
        values.append(count)

raw_term_doc_matrix = csr_matrix(
    (values, (row_indices, col_indices)),
    shape=(vocab_size, num_documents),
    dtype=np.float32
)

print(f"Matrix shape: {raw_term_doc_matrix.shape}")
print(f"Non-zero elements: {raw_term_doc_matrix.nnz:,}")
print(f"Sparsity: {100 * (1 - raw_term_doc_matrix.nnz / (vocab_size * num_documents)):.4f}%")

# ========================================
# APPLY TF-IDF TRANSFORMATION
# ========================================

print("\n[3/6] Applying TF-IDF transformation...")

# TF-IDF transformer
tfidf_transformer = TfidfTransformer(
    norm='l2',        # L2 normalization
    use_idf=True,     # Use IDF weighting
    smooth_idf=True,  # Add 1 to document frequencies (avoid division by zero)
    sublinear_tf=False  # Don't use log scaling for term frequency
)

# Transform the matrix
tfidf_matrix = tfidf_transformer.fit_transform(raw_term_doc_matrix)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"TF-IDF matrix type: {type(tfidf_matrix)}")

# ========================================
# COMPUTE SVD FOR BOTH MATRICES
# ========================================

print("\n[4/6] Computing SVD embeddings...")

# Best dimension from Task 4 (modify this based on your Task 4 results)
best_d = 100  # Change this to your best performing dimension

print(f"Using embedding dimension d={best_d}")

# --- RAW SVD ---
print(f"\n  Computing Raw SVD (d={best_d})...")
U_raw, sigma_raw, Vt_raw = svds(raw_term_doc_matrix, k=best_d)

# Reverse to get largest singular values first
U_raw = U_raw[:, ::-1]
sigma_raw = sigma_raw[::-1]
Vt_raw = Vt_raw[::-1, :]

# Generate embeddings
Sigma_raw = np.diag(sigma_raw)
embeddings_raw = U_raw @ Sigma_raw

print(f"    Raw embeddings shape: {embeddings_raw.shape}")
print(f"    Top 5 singular values: {sigma_raw[:5]}")

# --- TF-IDF SVD ---
print(f"\n  Computing TF-IDF SVD (d={best_d})...")
U_tfidf, sigma_tfidf, Vt_tfidf = svds(tfidf_matrix, k=best_d)

# Reverse
U_tfidf = U_tfidf[:, ::-1]
sigma_tfidf = sigma_tfidf[::-1]
Vt_tfidf = Vt_tfidf[::-1, :]

# Generate embeddings
Sigma_tfidf = np.diag(sigma_tfidf)
embeddings_tfidf = U_tfidf @ Sigma_tfidf

print(f"    TF-IDF embeddings shape: {embeddings_tfidf.shape}")
print(f"    Top 5 singular values: {sigma_tfidf[:5]}")

# ========================================
# QUALITY CHECK 1: NEAREST NEIGHBORS
# ========================================

print("\n[5/6] QUALITY CHECK 1: Nearest Neighbors Comparison")
print("="*60)

def find_nearest_neighbors(word, embeddings, vocabulary, word_to_idx, k=5):
    """Find k nearest neighbors using cosine similarity."""
    if word not in word_to_idx:
        return []
    
    word_idx = word_to_idx[word]
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    
    # Get word vector
    word_vec = normalized[word_idx:word_idx+1]
    
    # Compute similarities
    similarities = (normalized @ word_vec.T).flatten()
    
    # Get top k+1 (including the word itself)
    top_indices = np.argsort(similarities)[::-1][:k+1]
    
    # Filter out the word itself
    neighbors = []
    for idx in top_indices:
        if idx != word_idx:
            neighbors.append({
                'word': vocabulary[idx],
                'similarity': float(similarities[idx])
            })
            if len(neighbors) == k:
                break
    
    return neighbors

# Choose 5 diverse words for comparison
test_words = ['king', 'computer', 'play', 'good', 'year']

print("\nComparing top-5 neighbors for 5 diverse words:\n")

for word in test_words:
    if word not in word_to_idx:
        print(f"'{word}' not in vocabulary, skipping...")
        continue
    
    print(f"Word: '{word}'")
    print("-" * 50)
    
    # Raw SVD neighbors
    neighbors_raw = find_nearest_neighbors(word, embeddings_raw, vocabulary, word_to_idx, k=5)
    print("  Raw SVD:")
    for i, n in enumerate(neighbors_raw, 1):
        print(f"    {i}. {n['word']:<20} ({n['similarity']:.4f})")
    
    # TF-IDF SVD neighbors
    neighbors_tfidf = find_nearest_neighbors(word, embeddings_tfidf, vocabulary, word_to_idx, k=5)
    print("  TF-IDF SVD:")
    for i, n in enumerate(neighbors_tfidf, 1):
        print(f"    {i}. {n['word']:<20} ({n['similarity']:.4f})")
    
    # Calculate overlap
    raw_words = {n['word'] for n in neighbors_raw}
    tfidf_words = {n['word'] for n in neighbors_tfidf}
    overlap = raw_words & tfidf_words
    
    print(f"  Overlap: {len(overlap)}/5 words in common")
    if overlap:
        print(f"  Common words: {', '.join(overlap)}")
    print()

# ========================================
# SAVE EMBEDDINGS FOR TASK 4 MLP
# ========================================

print("[6/6] Saving TF-IDF embeddings for MLP training...")

# Save TF-IDF embeddings
np.save(f'tfidf_svd_embeddings_d{best_d}.npy', embeddings_tfidf)
print(f"  Saved: tfidf_svd_embeddings_d{best_d}.npy")

# Save vocabulary mapping
vocab_mapping = {
    'vocabulary': vocabulary,
    'word_to_idx': word_to_idx
}

with open('tfidf_vocabulary_mapping.pkl', 'wb') as f:
    pickle.dump(vocab_mapping, f)
print(f"  Saved: tfidf_vocabulary_mapping.pkl")

# Also save raw embeddings for comparison
np.save(f'raw_svd_embeddings_d{best_d}.npy', embeddings_raw)
print(f"  Saved: raw_svd_embeddings_d{best_d}.npy")

