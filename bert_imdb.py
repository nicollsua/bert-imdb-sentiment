# Run in Google Colab with GPU: Runtime > Change runtime type > T4 GPU
# Cell 1:
# !pip install -q transformers datasets accelerate scikit-learn gradio matplotlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# ---------- Data ----------
MODEL = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("stanfordnlp/imdb")  # 25k train / 25k test, labels 0=neg 1=pos


def tokenize(batch):
    # truncation keeps long reviews within BERT's limit; padding is done per-batch
    return tok(batch["text"], truncation=True, max_length=256)


ds = ds.map(tokenize, batched=True)
train_ds = ds["train"].shuffle(seed=42)
# To save time, use a 5k-review validation subset of the test split
val_ds = ds["test"].shuffle(seed=42).select(range(5000))
collator = DataCollatorWithPadding(tok)

# ---------- Model ----------
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


args = TrainingArguments(
    output_dir="bert-imdb",
    num_train_epochs=2,
    per_device_train_batch_size=16,  # lower to 8 if you hit CUDA out of memory
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ---------- Train + evaluate ----------
trainer.train()
print(trainer.evaluate())  # accuracy + F1 for the report
trainer.save_model("bert-imdb-final")
tok.save_pretrained("bert-imdb-final")

# ---------- Loss curves for the report ----------
log = trainer.state.log_history
train_pts = [(l["step"], l["loss"]) for l in log if "loss" in l]
eval_pts = [(l["step"], l["eval_loss"]) for l in log if "eval_loss" in l]
plt.plot(*zip(*train_pts), label="train loss")
plt.plot(*zip(*eval_pts), "o-", label="val loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# ---------- Demo ----------
import gradio as gr

demo_model = trainer.model.eval()


def predict(text):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(demo_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(demo_model(**inputs).logits, dim=-1)[0].cpu()
    return {"Negative": float(probs[0]), "Positive": float(probs[1])}


gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Paste a movie review"),
    outputs=gr.Label(num_top_classes=2),
    title="BERT Movie Review Sentiment",
).launch(share=True)  # share=True gives a public link you can open on any device
