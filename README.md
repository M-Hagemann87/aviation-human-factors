# Aviation Human Factor Incidents & Preventive Measures

A multi-label machine learning classifier that identifies human factor 
categories and preventive measures from NASA ASRS aviation incident reports.

## Overview
- **Model:** MiniLM (all-MiniLM-L6-v2) sentence embeddings + XGBoost Binary Relevance
- **Target:** 69 labels across 4 categories (human factors, contributing factors, 
  preventive measures, FAR part)
- **Dataset:** ~37,600 NASA ASRS incident reports (2005–2025)
- **Features:** 388-dim (384 MiniLM + 4 OHE FAR part columns)

## Results
- Macro F1 scores vary by target; human_factors achieves strongest signal
- Stratified 70/15/15 split on human_factors proxy key

## Deployed App
🤗 [Live demo on Hugging Face Spaces](https://huggingface.co/spaces/MatheusHagemann/aviation-human-factor-preventive-measures)

## Stack
Python · XGBoost · scikit-learn · sentence-transformers · pandas · Gradio
