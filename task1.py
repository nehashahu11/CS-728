import json
import os
import csv
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def buildIndex(words):
    '''
    This function takes a list of words and returns two dictionary.
    One word to idx and another idx to word
    '''
    word_idx = {word: i for i, word in enumerate(words)}
    idx_word = {i: word for i, word in enumerate(words)}
    return word_idx, idx_word

def buildSparseCooccurrence(data, word_idx, windowSize):
    '''
    Builds sparse co-occurrence counts directly:
    returns list of (i, j, count)
    '''
    cooc = defaultdict(int)

    for documents in data.values():
        for record in documents:
            words = record[1].split()
            indices = [word_idx[w] for w in words if w in word_idx]

            n = len(indices)
            for i in range(n):
                center = indices[i]
                left = max(0, i - windowSize)
                right = min(n, i + windowSize + 1)

                for j in range(left, right):
                    context = indices[j]
                    if center != context:
                        cooc[(center, context)] += 1

    pairs = [(i, j, c) for (i, j), c in cooc.items()]
    return pairs

class GloveModel(nn.Module):
    def __init__(self, vocabSize, embeddingDim):
        super().__init__()

        self.W1 = nn.Embedding(vocabSize, embeddingDim, sparse=True)
        self.W2 = nn.Embedding(vocabSize, embeddingDim, sparse=True)
        self.b1 = nn.Embedding(vocabSize, 1, sparse=True)
        self.b2 = nn.Embedding(vocabSize, 1, sparse=True)

        nn.init.uniform_(self.W1.weight, -0.5/embeddingDim, 0.5/embeddingDim)
        nn.init.uniform_(self.W2.weight, -0.5/embeddingDim, 0.5/embeddingDim)
        nn.init.zeros_(self.b1.weight)
        nn.init.zeros_(self.b2.weight)


def gloveLoss(model, I, J, X, xMax=100, alpha=0.75):
    w1 = model.W1(I)
    w2 = model.W2(J)

    b1 = model.b1(I).squeeze()
    b2 = model.b2(J).squeeze()

    dot = torch.sum(w1 * w2, dim=1)
    preds = dot + b1 + b2

    weights = torch.minimum(
        X / xMax,
        torch.ones_like(X)
    ) ** alpha

    return (weights * (preds - torch.log(X)) ** 2).mean()


def gloveEpochTorch(
    model,
    optimizer,
    I_all,
    J_all,
    X_all,
    batchSize=1048576,
    device="cpu"
):
    perm = torch.randperm(len(I_all), device=device)

    totalLoss = 0.0
    totalPairs = 0

    for start in range(0, len(I_all), batchSize):
        idx = perm[start:start + batchSize]

        I = I_all[idx]
        J = J_all[idx]
        X = X_all[idx]

        optimizer.zero_grad()
        loss = gloveLoss(model, I, J, X)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item() * len(idx)
        totalPairs += len(idx)

    return totalLoss / totalPairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)

    args = parser.parse_args()

    windowSize = args.window
    embeddingDim = args.dim
    numEpochs = args.epochs
    learningRate = args.lr

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = f"results/window_{windowSize}"
    os.makedirs(results_dir, exist_ok=True)

    pairs_path = f"{results_dir}/pairs.npy"

    with open("./dataset/updated_vocab_document_dict.json", "r") as f:
        data = json.load(f)

    words = list(data.keys())
    vocabSize = len(words)
    word_idx, idx_word = buildIndex(words)

    # ---- build or load sparse pairs ----
    if os.path.exists(pairs_path):
        print("Loading sparse co-occurrence pairs")
        pairs = np.load(pairs_path, allow_pickle=True).tolist()
    else:
        print("Building sparse co-occurrence pairs")
        pairs = buildSparseCooccurrence(data, word_idx, windowSize)
        np.save(pairs_path, np.array(pairs, dtype=object))

    # ---- move pairs to GPU ONCE ----
    I_all = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
    J_all = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
    X_all = torch.tensor([p[2] for p in pairs], dtype=torch.float, device=device)

    model = GloveModel(vocabSize, embeddingDim).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=learningRate)

    lossHistory = []

    print("=" * 60)
    print(f"Training | window={windowSize}, dim={embeddingDim}")
    start_time = datetime.now()
    print(f"Start time: {start_time}")

    for epoch in range(numEpochs):
        loss = gloveEpochTorch(
            model,
            optimizer,
            I_all,
            J_all,
            X_all,
            batchSize=1024,
            device=device
        )
        lossHistory.append(loss)

        if epoch % 5 == 0 or epoch == numEpochs - 1:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    end_time = datetime.now()
    print(f"End time: {end_time}")
    print(f"Total time: {end_time - start_time}")

    # ---- save loss ----
    with open(f"{results_dir}/loss_dim{embeddingDim}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(lossHistory):
            writer.writerow([i, l])

    # ---- plot loss ----
    plt.figure()
    plt.plot(lossHistory)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"GloVe Loss (window={windowSize}, dim={embeddingDim})")
    plt.grid(True)
    plt.savefig(f"{results_dir}/loss_dim{embeddingDim}.png")
    plt.close()

    # ---- save embeddings ----
    embeddings = (
        model.W1.weight.data +
        model.W2.weight.data
    ).cpu().numpy()

    np.save(f"{results_dir}/embeddings_dim{embeddingDim}.npy", embeddings)
   # ---- save embeddings ----
    embeddings = (
    model.W1.weight.data +
    model.W2.weight.data
).cpu().numpy()

    np.save(f"{results_dir}/embeddings_dim{embeddingDim}.npy", embeddings)

# ---- save vocab mappings ----
    with open(f"{results_dir}/word_to_idx.json", "w") as f:
        json.dump(word_idx, f)

    with open(f"{results_dir}/idx_to_word.json", "w") as f:
        json.dump(idx_word, f)

