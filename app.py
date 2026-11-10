# Standalone demo: loads the fine-tuned model and launches the Gradio interface.
# Usage: python app.py
# Set MODEL_PATH to the folder containing the saved model (config.json, model.safetensors, tokenizer files).
# In Colab with Google Drive mounted: "/content/drive/MyDrive/bert-imdb-final"

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "/content/drive/MyDrive/bert-imdb-final"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()


def predict(text):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu()
    return {"Negative": float(probs[0]), "Positive": float(probs[1])}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Paste a movie review"),
    outputs=gr.Label(num_top_classes=2),
    title="BERT Movie Review Sentiment",
)

if __name__ == "__main__":
    demo.launch(share=True)
