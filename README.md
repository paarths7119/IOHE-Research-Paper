# Drug-Drug Interaction Prediction using Machine Learning

## Overview

This project predicts whether two drugs will interact using machine learning
and molecular fingerprints generated from chemical structures.

The system uses:
- TWOSIDES FDA adverse event dataset
- RDKit Morgan fingerprints
- Cosine similarity
- Logistic Regression
- Random Forest
- XGBoost
- Neural Networks (PyTorch)

## Best Performance

Neural Network achieved:
- AUROC: 0.753

## Project Structure

```text
notebooks/      -> Training notebooks
plots/          -> Evaluation plots
app.py          -> Gradio demo
requirements.txt
```

## Technologies Used

- Python
- RDKit
- PyTorch
- XGBoost
- Scikit-learn
- Gradio

## Dataset

TWOSIDES dataset:
https://tatonettilab.org/resources/

## Run

```bash
pip install -r requirements.txt
python app.py
```