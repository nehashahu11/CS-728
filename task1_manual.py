import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import os


def buildIndex(words):
    '''
    This function takes a list of words and returns two dictionary. One word to idx and another idx to word
    '''
    word_idx = {word: i for i, word in enumerate(words)}
    idx_word = {i: word for i, word in enumerate(words)}
    return word_idx, idx_word


def updateCooccuranceMatrix(cooccuranceMatrix,windowSize,document,word_idx):
    '''
    This function updates a given co-occurance matrix provided document and window size
    '''
    indices = [word_idx[w] for w in document.split() if w in word_idx]

    n = len(indices)

    for i in range(n):
        center = indices[i]

        left = max(0, i - windowSize)
        right = min(n, i + windowSize + 1)

        for j in range(left, right):
            context = indices[j]
            cooccuranceMatrix[center][context] += 1

    return cooccuranceMatrix


def initializeGloveWeights(vocabSize, embeddingDim):
    '''
    This function initializes weights needed to calculate glove
    '''
    # Small random initialization
    W1 = np.random.uniform(
        low=-0.5 / embeddingDim,
        high=0.5 / embeddingDim,
        size=(vocabSize, embeddingDim)
    )

    W2 = np.random.uniform(
        low=-0.5 / embeddingDim,
        high=0.5 / embeddingDim,
        size=(vocabSize, embeddingDim)
    )

    # Bias terms initialized to zero
    b1 = np.zeros(vocabSize)
    b2 = np.zeros(vocabSize)

    return W1, W2, b1, b2


def getNonZeroPairs(cooccuranceMatrix):
    pairs = []
    V = cooccuranceMatrix.shape[0]
    for i in range(V):
        for j in range(V):
            if cooccuranceMatrix[i][j] > 0:
                pairs.append((i, j, cooccuranceMatrix[i][j]))

    return pairs


def initializeAdagrad(vocabSize, embeddingDim):
    eps = 1e-8

    G_W1 = np.zeros((vocabSize, embeddingDim))
    G_W2 = np.zeros((vocabSize, embeddingDim))
    G_b1 = np.zeros(vocabSize)
    G_b2 = np.zeros(vocabSize)

    return G_W1, G_W2, G_b1, G_b2, eps


def gloveSGDEpochAdagrad(
    pairs,
    W1, W2, b1, b2,
    G_W1, G_W2, G_b1, G_b2,
    learningRate=0.05,
    xMax=100,
    alpha=0.75,
    batchSize=512,
    eps=1e-8
):
    np.random.shuffle(pairs)

    totalLoss = 0.0
    totalPairs = 0

    for start in range(0, len(pairs), batchSize):
        batch = pairs[start:start + batchSize]

        I = np.array([p[0] for p in batch])
        J = np.array([p[1] for p in batch])
        X = np.array([p[2] for p in batch])

        weights = np.minimum(X / xMax, 1.0) ** alpha

        dot = np.sum(W1[I] * W2[J], axis=1)
        preds = dot + b1[I] + b2[J]

        errors = preds - np.log(X)

        totalLoss += np.sum(weights * (errors ** 2))
        totalPairs += len(batch)

        scale = 2 * weights * errors

        gradW1 = scale[:, None] * W2[J]
        gradW2 = scale[:, None] * W1[I]

        for k in range(len(I)):
            i = I[k]
            j = J[k]

            G_W1[i] += gradW1[k] ** 2
            G_W2[j] += gradW2[k] ** 2
            G_b1[i] += scale[k] ** 2
            G_b2[j] += scale[k] ** 2

            W1[i] -= learningRate * gradW1[k] / np.sqrt(G_W1[i] + eps)
            W2[j] -= learningRate * gradW2[k] / np.sqrt(G_W2[j] + eps)
            b1[i] -= learningRate * scale[k] / np.sqrt(G_b1[i] + eps)
            b2[j] -= learningRate * scale[k] / np.sqrt(G_b2[j] + eps)

    return totalLoss / totalPairs


if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    with open('./dataset/updated_vocab_document_dict.json', 'r') as file:
        data = json.load(file)

    words = list(data.keys())
    cooccuranceMatrix = [[0]* len(words) for _ in range(len(words))]

    word_idx,idx_word = buildIndex(words)

    print("creating co-occurance matrix")
    for documents in list(data.values()):
        for record in documents:
            cooccuranceMatrix = updateCooccuranceMatrix(
                cooccuranceMatrix,
                3,
                record[1],
                word_idx
            )

    cooccuranceMatrix = np.array(cooccuranceMatrix)
    pairs = getNonZeroPairs(cooccuranceMatrix)

    embedding_dims = [50, 100, 200, 300]
    numEpochs = 50
    learningRate = 0.05

    for dim in embedding_dims:

        print("=" * 60)
        print(f"Starting training | dim={dim}, epochs={numEpochs}")

        start_time = datetime.now()
        print(f"Start time: {start_time}")

        W1, W2, b1, b2 = initializeGloveWeights(len(words), dim)
        G_W1, G_W2, G_b1, G_b2, eps = initializeAdagrad(len(words), dim)

        lossHistory = []

        for epoch in range(numEpochs):
            loss = gloveSGDEpochAdagrad(
                pairs,
                W1, W2, b1, b2,
                G_W1, G_W2, G_b1, G_b2,
                learningRate=learningRate,
                xMax=100,
                alpha=0.75,
                batchSize=512,
                eps=eps
            )
            lossHistory.append(loss)

            if epoch % 10 == 0 or epoch == numEpochs - 1:
                print(f"Epoch {epoch}: loss = {loss:.4f}")

        end_time = datetime.now()
        print(f"End time: {end_time}")
        print(f"Total training time: {end_time - start_time}")

        # save loss
        csv_path = f"results/loss_dim{dim}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, l in enumerate(lossHistory):
                writer.writerow([i, l])

        # plot loss
        plt.figure()
        plt.plot(lossHistory)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"GloVe Loss (dim={dim}, epochs={numEpochs})")
        plt.grid(True)
        plt.savefig(f"results/loss_dim{dim}.png")
        plt.close()
