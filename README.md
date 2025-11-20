# LLM Prompt Expansion Evaluation using CLIP Score

A comparative study of Large Language Models (LLMs) for automatic prompt expansion, evaluated using CLIP Score as a semantic coherence metric.

## Overview

This project evaluates and compares four state-of-the-art open-source LLMs (Mistral, Phi3, Llama3.2, and Gemma2) in their ability to expand simple class names into detailed, semantically coherent descriptions suitable for text-to-image generation systems like Stable Diffusion.

The evaluation is performed on 1,000 ImageNet classes using CLIP Score as a proxy metric for text-image semantic alignment.

## Key Features

- **Automated Prompt Expansion**: Generates detailed descriptions from simple class names
- **Quantitative Evaluation**: Uses CLIP Score to measure semantic coherence
- **Comprehensive Analysis**: Tests across 1,000 ImageNet classes
- **Performance Metrics**: Evaluates both quality (CLIP Score) and speed (response time)
- **Rich Visualizations**: Generates distributions, cumulative plots, and Excel reports

## Quick Start
### Installation

```bash
git clone https://github.com/Deste-user/LLM_Coherence_Expanded_Prompt.git
cd LLM_Coherence_Expanded_Prompt
pip install -r requirements.txt
```

### Running the Experiment

```bash
python main.py
```

The script will:
1. Download ImageNet-1000 dataset (if not present)
2. Start Ollama server
3. Download required LLM models
4. Generate image embeddings (cached)
5. Generate prompt expansions for all models
6. Calculate CLIP Scores
7. Generate visualizations and Excel reports

## Project Structure

```
.
├── main.py                      # Main orchestrator
├── env_for_chatting.py          # Ollama communication & prompt engineering
├── analysis.py                  # CLIP embedding & scoring
├── grafic_analysis.py           # Visualization generation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── Relazione Progetto/           # It contains all sources of the Report LaTeX and the pdf
├── Presentazione - Valutazione della Coerenza Semantica tramite ClipScore.pdf #Presentation pdf
│
│
│
├── Small-ImageNet-Validation-Dataset-1000-Classes/  # Dataset (auto-downloaded)
├── images_embedding/            # Cached image embeddings
├── ClipScores.xlsx          # Detailed score table
├── avg_response_time.png    # Response time comparison
├── classes_below_threshold.png
├── Distributions/           # Score distribution plots
└── CumulativeDistribution/  # Cumulative distribution plots
```

## Configuration

### Models Tested

Modify the model list in `main.py`:

```python
models = ["mistral:latest", "phi3:latest", "llama3.2:latest", "gemma2:latest"]
```

### Parameters

Key parameters in `main.py`:

```python
NUM_CLASS = 1000           # Number of ImageNet classes
NUM_IMG_4_CLASS = 5        # Images per class
num_prompts = 3            # Expansions per class per model
max_length = 77            # CLIP token limit
```

## Methodology

The system uses few-shot learning with 13 carefully crafted examples to guide LLMs in generating realistic, detailed descriptions:

```
Short: "a portrait of a blonde woman"
Long: "portrait of a pretty blonde woman, a flower crown, earthy makeup, 
flowing maxi dress with colorful patterns and fringe, a sunset or nature 
scene, green and gold color scheme"
```

## Visualizations

The project generates:

- **Distribution Histograms**: Score distribution in 10 bins (0.0-1.0)
- **Cumulative Plots**: Number of classes ≤ threshold
- **Response Time Bar Chart**: Average time per model
- **Threshold Analysis**: Classes below global average
- **Excel Report**: Detailed scores with conditional formatting


