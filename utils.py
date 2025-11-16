import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


# ============================== TESTS UNITAIRES QUE LE PROF DEMANDE ==============================

def encode_sentence_to_onehot(sentence, vocab, max_len):
    """
    Encode une phrase brute en tenseur one-hot (1, T, V) avec PAD silencieux,
    et retourne aussi la longueur réelle (1,).
    """
    toks = sentence.lower().split()
    ids = [vocab.get(w, vocab["<UNK>"]) for w in toks]
    true_len = min(len(ids), max_len)

    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))

    V = len(vocab)
    one_hot = np.zeros((max_len, V), dtype=np.float32)
    pad_id = vocab["<PAD>"]
    for i, idx_word in enumerate(ids):
        if idx_word != pad_id:
            one_hot[i, idx_word] = 1.0

    X = torch.tensor(one_hot).unsqueeze(0)
    L = torch.tensor([true_len], dtype=torch.long)
    return X, L


def test_one_word(model, vocab, device, label2id, max_len):
    candidate_words = [w for w in vocab.keys() if w not in ("<PAD>", "<UNK>")]
    if not candidate_words:
        print("Aucun mot dans le vocab (hors <PAD>/<UNK>).")
        return
    word = candidate_words[0]
    X, L = encode_sentence_to_onehot(word, vocab, max_len)
    X, L = X.to(device), L.to(device)
    with torch.no_grad():
        logits = model(X, L)
        pred_id = logits.argmax(dim=1).item()
    print(f"[ONE WORD] mot='{word}' -> logits.shape={tuple(logits.shape)} ; pred_id={pred_id}")


def test_short_sequence(model, vocab, device, max_len):
    words = [w for w in vocab.keys() if w not in ("<PAD>", "<UNK>")][:3]
    if len(words) < 3:
        print("Pas assez de mots pour un test 3 tokens.")
        return
    sent = " ".join(words)
    X, L = encode_sentence_to_onehot(sent, vocab, max_len)
    X, L = X.to(device), L.to(device)
    with torch.no_grad():
        logits = model(X, L)
    print(f"[SHORT SEQ] phrase='{sent}' -> X.shape={tuple(X.shape)} ; logits.shape={tuple(logits.shape)}")


def test_with_dataloader(model, loader, device):
    model.eval()
    for batch in loader:
        if len(batch) == 3:
            X, y, L = batch
        else:
            X, y = batch
            L = torch.full((X.size(0),), X.size(1), dtype=torch.long)

        X, y, L = X.to(device), y.to(device), L.to(device)
        with torch.no_grad():
            logits = model(X, L)
            preds = logits.argmax(dim=1)
        print(f"[DATALOADER] batch X={tuple(X.shape)} ; logits={tuple(logits.shape)} ; "
              f"acc_batch={(preds==y).float().mean().item()*100:.2f}%")
        break


def predict_sentence(model, sentence, vocab, label2id, device, max_len):
    id2label = {i: l for l, i in label2id.items()}
    X, L = encode_sentence_to_onehot(sentence, vocab, max_len)
    X, L = X.to(device), L.to(device)
    with torch.no_grad():
        logits = model(X, L)
        pred_id = logits.argmax(dim=1).item()
        pred_label = id2label[pred_id]
    return pred_label

# ============================== POUR FAIRE DES PLOT ==============================


def plot_learning_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker='s')
    plt.title("Courbes de Loss")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color='green', marker='^')
    plt.title("Courbe d'Accuracy (Validation)")
    plt.xlabel("Époques")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_and_plot_confusion(model, loader, label2id, device, normalize=True, title="Matrice de confusion (test)"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, L in loader:
            X, y, L = X.to(device), y.to(device), L.to(device)
            logits = model(X, L)
            preds = logits.argmax(1)
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    id2label = {i: l for l, i in label2id.items()}
    labels_order = [id2label[i] for i in range(len(id2label))]

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels_order)))
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.xticks(range(len(labels_order)), labels_order, rotation=45, ha="right")
    plt.yticks(range(len(labels_order)), labels_order)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]:.2f}" if normalize else str(int(cm[i,j]))
            plt.text(j, i, txt, ha='center', va='center')
    plt.tight_layout()
    plt.show()

    print("\nClassification report :")
    print(classification_report(y_true, y_pred, target_names=labels_order))


def plot_word_distribution(texts, top_k=20, lowercase=True, stopwords=None,
                           min_freq=None, title="Distribution des mots les plus fréquents"):
    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)

    counter = Counter()
    for text in texts:
        toks = text.split()
        if lowercase:
            toks = [t.lower() for t in toks]
        for w in toks:
            if w in stopwords:
                continue
            counter[w] += 1

    if min_freq is not None and min_freq > 1:
        counter = Counter({w: c for w, c in counter.items() if c >= min_freq})

    if len(counter) == 0:
        print("Aucun mot à afficher.")
        return []

    most_common = counter.most_common(top_k)
    words, freqs = zip(*most_common)

    plt.figure(figsize=(10, 5))
    plt.bar(words, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Mots")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()

    return most_common


def plot_label_distribution(labels, title="Distribution des classes (labels)"):
    counter = Counter(labels)
    classes, freqs = zip(*sorted(counter.items(), key=lambda x: x[0]))
    plt.figure(figsize=(7, 4))
    plt.bar(classes, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.tight_layout()
    plt.show()
    return dict(counter)

# ============================== STATS DES DONNÉES ==============================

def lengths_stats(texts, max_len=30):
    lens = [len(s.lower().split()) for s in texts]
    L = np.array(lens)
    trunc_pct = (L > max_len).mean() * 100
    return {
        "n": len(L),
        "mean": float(L.mean()),
        "median": float(np.median(L)),
        "max": int(L.max()),
        "truncated_%": trunc_pct
    }


def oov_rate(texts, vocab):
    total = 0
    oov = 0
    for s in texts:
        for w in s.lower().split():
            total += 1
            if w not in vocab:
                oov += 1
    return (oov / max(total, 1)) * 100


# ============================== POUR LA PARTIE AMÉLIARATION  ==============================


def compute_class_weights(labels, label2id, device):
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(label2id)
    weights = torch.zeros(n_classes, dtype=torch.float32)
    for lab, idx in label2id.items():
        freq = counts[lab] / total
        weights[idx] = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * n_classes
    return weights.to(device)


def visualize_embeddings(model, vocab, top_k=200):
    with torch.no_grad():
        E = model.i2e.weight.detach().cpu().T
    words = list(vocab.keys())[:top_k]
    X = E[:top_k, :]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    for i, w in enumerate(words):
        plt.text(coords[i, 0], coords[i, 1], w, fontsize=8, alpha=0.7)
    plt.title("Projection PCA des embeddings de mots (top 200)")
    plt.show()

