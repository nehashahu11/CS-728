import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
import json

########################################
# Constants
########################################

PAD_LABEL = -100

########################################
# Dataset (sentence-level, OOV → zero)
########################################

class NERSentenceDataset(Dataset):
    def __init__(self, split, word_to_idx, embeddings):
        self.data = split
        self.word_to_idx = word_to_idx
        self.embeddings = embeddings
        self.dim = embeddings.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        embs = []

        for tok in sent["tokens"]:
            tok_l = tok.lower()
            if tok_l in self.word_to_idx:
                embs.append(self.embeddings[self.word_to_idx[tok_l]])
            else:
                embs.append(np.zeros(self.dim, dtype=np.float32))  # OOV → zero

        return (
            torch.tensor(embs, dtype=torch.float32),
            torch.tensor(sent["ner_tags"], dtype=torch.long)
        )


########################################
# Collate function (pad + flatten)
########################################

def collate_sentences(batch):
    batch_size = len(batch)
    max_len = max(len(x[0]) for x in batch)
    dim = batch[0][0].shape[1]

    X = torch.zeros(batch_size, max_len, dim)
    y = torch.full((batch_size, max_len), PAD_LABEL, dtype=torch.long)

    for i, (embs, labels) in enumerate(batch):
        L = len(labels)
        X[i, :L] = embs
        y[i, :L] = labels

    return X.view(-1, dim), y.view(-1)


########################################
# Model
########################################

class TokenMLP(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        return self.net(x)


########################################
# Training + Evaluation
########################################

def train_and_eval(embeddings, word_to_idx, tag_names, epochs=25, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
    train_ds = NERSentenceDataset(dataset["train"], word_to_idx, embeddings)
    test_ds  = NERSentenceDataset(dataset["test"], word_to_idx, embeddings)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sentences
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sentences
    )

    num_labels = len(tag_names)
    model = TokenMLP(embeddings.shape[1], num_labels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)

    # -------- TRAIN --------
    for epoch in range(epochs):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask = (y != PAD_LABEL)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/total_tokens:.4f}")

    # -------- EVAL --------
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            mask = (y != PAD_LABEL)

            y_true.extend(y[mask].tolist())
            y_pred.extend(preds[mask].tolist())

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=tag_names, digits=4))
    print(f"Macro F1 = {f1_score(y_true, y_pred, average='macro'):.4f}")


########################################
# Main: Train RAW SVD for all 4 dims
########################################

if __name__ == "__main__":

    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
    tag_names = dataset["train"].features["ner_tags"].feature.names

    svd_dims = [50, 100, 200, 300]

    for dim in svd_dims:
        print("\n" + "=" * 60)
        print(f"Training RAW SVD embeddings (dim={dim})")
        print("=" * 60)

        emb_path = f"results/dim_{dim}/embeddings_dim{dim}.npy"
        vocab_path = f"results/dim_{dim}/vocab.json"

        embeddings = np.load(emb_path)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        word_to_idx = {w: i for i, w in enumerate(vocab)}

        # Sanity check
        assert embeddings.shape[0] == len(word_to_idx), \
            f"Mismatch for dim={dim}"

        train_and_eval(
            embeddings,
            word_to_idx,
            tag_names,
            epochs=25
        )
