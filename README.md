# Domain-Adapted Text Summarisation using T5 and LoRA

This repository contains an MSc-level Natural Language Processing project focused on domain-adapted abstractive text summarisation. The system fine-tunes a T5-base transformer using parameter-efficient fine-tuning (LoRA) on educational institution texts derived from the WikiAsp dataset.

## Project Overview
Generic summarisation models often struggle with long, structured, domain-specific documents. This project addresses that limitation by adapting a pretrained T5 model to the educational institution domain and deploying it as a web application.

## Key Features
- Domain-specific summarisation using WikiAsp (educational institutions)
- Parameter-efficient fine-tuning with LoRA
- Transformer-based abstractive summarisation (T5-base)
- ROUGE-based evaluation with human qualitative analysis
- Flask web application for real-time inference

## Technologies Used
- Python
- Hugging Face Transformers
- PEFT (LoRA)
- PyTorch
- Datasets
- Flask

## Repository Structure
- `notebooks/` – Model training and experimentation
- `training/` – Training and evaluation scripts
- `app.py` – Flask web application
- `report/` – Final MSc project report
- `data/` – Dataset description (no raw data included)
- `models/` – Model usage notes (weights not uploaded)

## Dataset
The project uses the WikiAsp dataset (educational institution domain). Due to licensing and size constraints, the dataset is not included. Please refer to the original dataset source.

## Results Summary
- ROUGE-1: ~18
- ROUGE-2: ~3
- ROUGE-L: ~15
Human evaluation showed strong fluency, coherence, and factual grounding.

## How to Run the App
```bash
pip install -r requirements.txt
python app.py
