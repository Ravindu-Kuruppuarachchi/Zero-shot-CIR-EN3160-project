# StyleNStay: Zero-Shot Composed Image Retrieval for Fashion

This project implements and extends concepts from the paper **"Zero-shot Composed Text-Image Retrieval"** to create **StyleNStay**, an interactive web application for fashion discovery. It allows users to search for clothing items using a reference image and natural language modifications, leveraging a specialized `TransAgg` model trained for the fashion domain.

## Overview

**Composed Image Retrieval (CIR)** aims to find images matching a reference image modified according to a text description. This project focuses on a **zero-shot** approach, training on automatically generated data (`Laion-CIR-Template`, 'Laion-CIR-LLM', 'Laion-CIR-Combined') and evaluating on the `FashionIQ` benchmark without specific fine-tuning on it.

## Core Concept: Composed Image Retrieval (CIR)

Composed Image Retrieval is the task of finding a specific target image using two pieces of information:

- A reference image
- A relative text caption describing the desired modifications

For example: [reference image] + "but with short sleeves" → Should retrieve → [target image]

This project focuses on a **zero-shot approach**, where we train the model on a general, automatically generated dataset (Laion-CIR) and then evaluate its performance on a specialized, human-annotated dataset (FashionIQ) without any fine-tuning on the evaluation set.

## Project Workflow

### 1. Training

We trained the `TransAgg` model (using a BLIP backbone) on the complete `Laion-CIR-Combined` dataset, which comprises all 32,000 triplets from the combined `Laion-CIR-Template` and `Laion-CIR-LLM` datasets. This extensive training allows the model to learn robust zero-shot generalization.

-   **Model:** The `TransAgg` architecture with a BLIP backbone was used.
-   **Dataset:** `Laion-CIR-Combined` (incorporating `Laion-CIR-Template` and `Laion-CIR-LLM`).
-   **Configuration:** All training parameters (batch size, learning rate, epochs, file paths) were managed via `config.py`.
-   **Validation:** During training, the model's performance was validated against the standard `FashionIQ` validation set after each epoch. The best-performing checkpoint was saved.

### 2. Evaluation

After training, we evaluated the final model's performance by running it on the FashionIQ validation set in an evaluation-only mode. This involved modifying the `main.py` script to load our trained checkpoint and execute the `trainer.eval_fiq()` function, which calculates the final Recall@10 and Recall@50 metrics.

## ✨ Application Showcase: StyleNStay ✨

StyleNStay demonstrates the practical application of the trained model. Users can browse products, select an item, and describe modifications to find similar items matching their specific criteria.

**Main Interface:**
<div align="center">
 
![User Interface](Images/UI(1).png)

</div>


* Displays product listings with dynamic pricing, ratings, badges, and wishlist/cart buttons.
* Allows filtering by category (Shirts, Dresses, etc.).
* Includes sorting options (Latest, Price, Rating).

**Visual Search Modal:**
<div align="center">
 
![User Interface](Images/UI(2).png)

</div>
* Activated by clicking "Find Similar" on a product.
* Shows the selected reference image.
* Provides a text box for users to enter modifications (e.g., "make it sleeveless", "change color to blue").
* Initiates the visual search using the trained `TransAgg` model.

## How to Use This Repository

### 1. Setup

#### a. Environment

Install all required Python packages:

```bash
pip install -r requirements.txt
```

If you have a CUDA-enabled GPU, make sure to install the correct version of PyTorch.

#### b. Download Datasets

- **FashionIQ:** Download from the official repository and place it in a known datasets directory
- **Laion-CIR-Template:** Download the images and `laion_template_info.json` file from the Google Drive link in the original README.md. Place them in their respective folders

#### c. Download Pre-trained Model

Download the base BLIP model (`model_base_retrieval_coco.pth`) required by the architecture.

#### d. Configure Paths

Before running anything, update the hardcoded paths in the `data/*.py` files and `model/model.py` to point to the correct locations of your datasets and the base BLIP model.

### 2. Data Curation (Optional)

To recreate the training subset:

```bash
# Run the filtering script
python Selectkeywords.py

# Run the sampling script
python random_sampler.py
```

### 3. Training a New Model

1. **Modify Data Loader:** Edit `data/laion_dataset_template.py` to load your training subset (e.g., `fashion_train_subset_2000.json`)

2. **Configure `config.py`:**
   - Set `laion_type: 'laion_template'` and `dataset: 'fiq'`
   - Set `comment` to a unique name for your training run
   - Set `save_path_prefix` to a folder where you want to save checkpoints
   - Adjust `batch_size`, `num_epochs`, etc. as needed

3. **Run Training:**
   ```bash
   python main.py
   ```

### 4. Evaluating a Trained Model

1. **Configure `config.py`:**
   - Set `dataset: 'fiq'` and ensure other parameters like `model_name` match the model you are evaluating

2. **Modify `main.py`:**
   - Comment out the `trainer.train()` line
   - Add the evaluation code block to load your checkpoint and call `trainer.eval_fiq()`
   - Make sure the `model_path` variable in the script points to your trained `.pth` file

3. **Run Evaluation:**
   ```bash
   python main.py
   ```

The final Recall scores will be printed to the console.

## Acknowledgements

This work is based on the official implementation of the paper "Zero-shot Composed Text-Image Retrieval" by Yikun Liu, Jiangchao Yao, Ya Zhang, Yanfeng Wang, and Weidi Xie.

