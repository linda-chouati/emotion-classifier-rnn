import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import torch.nn as nn
import torch.optim as optim

from config import MIN_FREQ, SEP, device

# -----------------------------------------------------------------
def set_seed(seed=42):
    """
    Fixe toutes les seeds utiles pour rendre les runs plus stables.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------

class OneHotDataset(Dataset):
    """
    class qui sert à :
        - charger les données depuis un fichier
        - transformer les phrases en tenseurs one hot
        - fournir les labels 
    """
    def __init__(self, filepath, max_len=20, min_freq=MIN_FREQ, sep=SEP,
                 vocab=None, label2id=None, build_on_texts=False):
        """
        Chargement des donnes brutes depuis un chemin 
        Construction ou bien reutilisation d un vacob et d un mapping label->id : 
            - Si vocab ou label2id sont fournis -> on les réutilise -> pas de rebuild
            - Sinon si build_on_texts=True, on le construit à partir de ce fichier (c moche mais avait des probs)
        """
        self.texts, self.labels = self._load_file(filepath, sep)
        self.max_len = max_len

        if vocab is None:
            if not build_on_texts:
                raise ValueError("vocab=None mais build_on_texts=False : passe un vocab commun ou mets build_on_texts=True pour ce split.")
            self.vocab = self._build_vocab(self.texts, min_freq)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)

        if label2id is None:
            unique_labels = sorted(set(self.labels))
            self.label2id = {lab: i for i, lab in enumerate(unique_labels)}
        else:
            self.label2id = label2id

    def _load_file(self, filepath, sep):
        texts, labels = [], []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) != 2:
                    continue
                texts.append(parts[0].strip())
                labels.append(parts[1].strip())
        return texts, labels

    def _build_vocab(self, texts, min_freq):
        # partie qui compte combien de fois chaque mot apparant 
        counter = Counter()
        for phrase in texts:
            for word in phrase.lower().split():
                counter[word] += 1

        vocab = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
        return vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, indice):
        sentence = self.texts[indice].lower().split()
        label = self.labels[indice]
    
        ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in sentence]
        true_len = min(len(ids), self.max_len)
    
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
    
        one_hot = np.zeros((self.max_len, self.vocab_size), dtype=np.float32)
        pad_id = self.vocab["<PAD>"]
        for i, idx_word in enumerate(ids):
            if idx_word != pad_id:
                one_hot[i, idx_word] = 1.0
    
        X = torch.tensor(one_hot)
        y = torch.tensor(self.label2id[label])
        L = torch.tensor(true_len, dtype=torch.long)
        return X, y, L


class RNNManual(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size=128, dropout_p=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # cocuhe linéaire qui transforme le one hot d un mot en un vecteur dense
        self.i2e = nn.Linear(input_size, emb_size, bias=False)
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            input_t = x[:, t, :]
            new_word = self.i2e(input_t)
            combined = torch.cat((new_word, hidden), dim=1)
            new_hidden = torch.tanh(self.i2h(combined))
            new_hidden = self.dropout(new_hidden)
            m = (t < lengths).float().unsqueeze(1)
            hidden = m * new_hidden + (1.0 - m) * hidden

        output = self.h2o(hidden)
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        if len(batch) == 3:
            inputs, labels, lengths = batch
        else:
            inputs, labels = batch
            lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)

        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, lengths = batch
            else:
                inputs, labels = batch
                lengths = torch.full((inputs.size(0),), inputs.size(1), dtype=torch.long)

            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0.0
    return running_loss / len(dataloader), accuracy


def train_full(model, train_loader, val_loader, epochs, lr, device,
               use_val_acc=True, seed=42, log_path=None, best_model_path=None,
               class_weights=None, dropout_tag=None, hidden_tag=None):
    """
    Entraîne le modèle et garde le meilleur en mémoire.
    Retourne l'historique des pertes et précisions.
    class_weights : Tensor pour CrossEntropyLoss(weight=...)
    dropout_tag / hidden_tag : juste pour logger proprement dans le CSV.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5
    )

    best_metric = -1.0 if use_val_acc else float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        need_header = (not os.path.exists(log_path)) or (os.path.getsize(log_path) == 0)
        if need_header:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("seed,lr,dropout,hidden,epoch,train_loss,val_loss,val_acc\n")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        metric_now = val_acc if use_val_acc else -val_loss
        if metric_now > best_metric:
            best_metric = metric_now
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if best_model_path is not None:
                torch.save(best_state, best_model_path)

        if log_path is not None:
            dr = dropout_tag if dropout_tag is not None else "-"
            hs = hidden_tag if hidden_tag is not None else "-"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{seed},{lr},{dr},{hs},{epoch+1},{train_loss:.6f},{val_loss:.6f},{val_acc:.4f}\n")

        print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Val acc {val_acc:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best in-memory model.")
    return history



def build_model(input_size, hidden_size, output_size, emb_size, dropout):
    """Réinstancie un modèle tout neuf à chaque run (important !)."""
    return RNNManual(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        emb_size=emb_size,
        dropout_p=dropout
    ).to(device)
