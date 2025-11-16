import importlib
import torch
from torch.utils.data import DataLoader

import feeling
import utils
from config import (
    data_train, data_val, data_test,
    MODE, EPOCHS, LR, BATCH_SIZE, MAX_LEN, MIN_FREQ,
    HIDDEN_SIZE, EMB_SIZE, DROPOUT, SEP, device
)

# Rechargement 
importlib.reload(utils)
importlib.reload(feeling)


# ---------- Préparation commune (datasets, loaders, modèle, poids de classes) ----------
def prepare_data_and_model():
    """
    Prépare vocab, labels, datasets, dataloaders, modèle et class_weights.
    Retourne un dict 'ctx' avec tout ce qu'il faut pour les fonctions de mode.
    """
    # Seed fixe pour stabilité
    feeling.set_seed(42)

    # Vocab & labels construits sur TRAIN
    train_tmp = feeling.OneHotDataset(
        data_train, max_len=MAX_LEN, min_freq=MIN_FREQ, sep=SEP,
        vocab=None, label2id=None, build_on_texts=True
    )
    shared_vocab = train_tmp.vocab
    shared_label2id = train_tmp.label2id
    print("Chargement ok (vocab et labels partagés)")
    print("vocab_size =", len(shared_vocab), "| n_labels =", len(shared_label2id))

    # Datasets / loaders
    train_dataset = feeling.OneHotDataset(data_train, max_len=MAX_LEN, sep=SEP, vocab=shared_vocab, label2id=shared_label2id)
    val_dataset   = feeling.OneHotDataset(data_val,   max_len=MAX_LEN, sep=SEP, vocab=shared_vocab, label2id=shared_label2id)
    test_dataset  = feeling.OneHotDataset(data_test,  max_len=MAX_LEN, sep=SEP, vocab=shared_vocab, label2id=shared_label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Poids de classes (déséquilibre)
    class_weights = utils.compute_class_weights(train_dataset.labels, shared_label2id, device)

    # Modèle
    model = feeling.RNNManual(
        input_size=len(shared_vocab),
        hidden_size=HIDDEN_SIZE,
        output_size=len(shared_label2id),
        emb_size=EMB_SIZE,
        dropout_p=DROPOUT
    ).to(device)

    return {
        "shared_vocab": shared_vocab,
        "shared_label2id": shared_label2id,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "model": model,
    }


# ------------------------------ 1) MODE TRAIN ------------------------------
def run_train_mode(ctx):
    print("\n== TRAIN ==")

    # Petites analyses -> plots de préparation des données
    utils.plot_word_distribution(ctx["train_dataset"].texts, top_k=20)
    utils.plot_label_distribution(ctx["train_dataset"].labels)
    print("== Longueurs ==")
    print("train:", utils.lengths_stats(ctx["train_dataset"].texts, max_len=MAX_LEN))
    print("val  :", utils.lengths_stats(ctx["val_dataset"].texts,   max_len=MAX_LEN))
    print("test :", utils.lengths_stats(ctx["test_dataset"].texts,  max_len=MAX_LEN))

    print("== OOV (% de mots hors vocab train) ==")
    print(f"val  : {utils.oov_rate(ctx['val_dataset'].texts,  ctx['train_dataset'].vocab):.2f}%")
    print(f"test : {utils.oov_rate(ctx['test_dataset'].texts, ctx['train_dataset'].vocab):.2f}%")

    # Entraînement
    history = feeling.train_full(
        ctx["model"], ctx["train_loader"], ctx["val_loader"],
        epochs=EPOCHS, lr=LR, device=device,
        use_val_acc=True, seed=42,
        class_weights=ctx["class_weights"],
        dropout_tag=DROPOUT, hidden_tag=HIDDEN_SIZE
    )
    utils.plot_learning_curves(history)

    # Éval finale + matrice de confusion + visu embeddings
    test_loss, test_acc = feeling.eval_epoch(ctx["model"], ctx["test_loader"], torch.nn.CrossEntropyLoss(), device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
    utils.evaluate_and_plot_confusion(ctx["model"], ctx["test_loader"], ctx["shared_label2id"], device)
    utils.visualize_embeddings(ctx["model"], ctx["shared_vocab"])


# ------------------------------ 2) MODE ONEWORD ------------------------------
def run_oneword_mode(ctx):
    print("\n== TEST 1 : un mot (B=1, T=1) ==")
    utils.test_one_word(ctx["model"], ctx["shared_vocab"], device, ctx["shared_label2id"], MAX_LEN)


# ------------------------------ 3) MODE SHORTSEQ ------------------------------
def run_shortseq_mode(ctx):
    print("\n== TEST 2 : séquence courte (B=1, T=3) ==")
    utils.test_short_sequence(ctx["model"], ctx["shared_vocab"], device, MAX_LEN)


# ------------------------------ 4) MODE DATALOADER ------------------------------
def run_dataloader_mode(ctx):
    print("\n== TEST 3 : DataLoader (batch > 1) ==")
    utils.test_with_dataloader(ctx["model"], ctx["val_loader"], device)


# ------------------------------ MAIN  ------------------------------
def main():
    ctx = prepare_data_and_model()

    if MODE == "train":
        run_train_mode(ctx)
    elif MODE == "oneword":
        run_oneword_mode(ctx)
    elif MODE == "shortseq":
        run_shortseq_mode(ctx)
    elif MODE == "dataloader":
        run_dataloader_mode(ctx)
    else:
        print(f"MODE inconnu: {MODE}. Choisis parmi: train | oneword | shortseq | dataloader")
