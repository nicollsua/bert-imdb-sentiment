# Sentiment Analysis on Movie Reviews: Fine-Tuning BERT

CSCI 390: Natural Language Processing and Applications (Dr. Qi Li)

**Team:** Somtochukwu Chukwudi-Edoka, Nicoll Suarez

## Overview

This project fine-tunes `bert-base-uncased` to classify IMDB movie reviews as **positive** or **negative**. It covers the full pipeline: loading data, tokenization, fine-tuning, evaluation, and a live web demo where you can paste in any review and get a prediction with a confidence score.

## Dataset

[IMDB](https://huggingface.co/datasets/stanfordnlp/imdb): 50,000 polar movie reviews, split evenly into 25,000 for training and 25,000 for testing. Labels: 0 = negative, 1 = positive.

## Model

- Base model: `bert-base-uncased`
- A linear classification head is added on top of BERT's pooled output
- All BERT weights are fine-tuned together with the classification head
- Built with the Hugging Face `transformers` `Trainer`, adapted from the official text-classification example

## Setup and Training

Run in Google Colab with a GPU (Runtime > Change runtime type > T4 GPU).

```bash
pip install transformers datasets accelerate scikit-learn gradio matplotlib
python bert_imdb.py
```

| Setting | Value |
|---|---|
| Epochs | 2 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max sequence length | 256 tokens |
| Precision | fp16 |

## Results

Evaluated on a 5,000-review subset of the test split.

| Metric | Score |
|---|---|
| Accuracy | 0.921 |
| F1 | 0.922 |

The training and validation loss curves are in `loss_curve.png`.

## Demo

The last part of the script launches a Gradio web app. Paste a movie review into the text box and the model returns the predicted sentiment with confidence scores for both classes. Running the script prints a public link you can open in any browser.

## Files

- `bert_imdb.py`: data loading, training, evaluation, loss plot, and demo
- `loss_curve.png`: training and validation loss
- `README.md`: this file

## Challenges

- **CUDA out of memory:** [describe what happened and how you fixed it, e.g. lowered the batch size]
- **Padding and truncation:** reviews vary a lot in length, so we truncate to 256 tokens and pad dynamically per batch.
- **Dataset loading error:** newer Hugging Face versions require the full dataset name (`stanfordnlp/imdb`).
