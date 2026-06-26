# NCRDLLM

**English** | [简体中文](README.zh-CN.md)

## Welcome to NCRDLLM

NCRDLLM (Prediction of ncRNA–Drug Response Associations Based on a Large Language Model) is a unified
framework for predicting associations between three types of non-coding RNA (circRNA, miRNA, lncRNA) and
drugs. The model constructs three groups of multimodal features — sequence features from the pretrained
foundation models RNA-FM and ChemBERTa, structural features from RNA secondary-structure graphs and drug
fingerprint/graph representations, and disease **semantic-similarity** features — and maps
them into the hidden space of **LLaMA-3.2-3B** through adapter modules, with **LoRA** used for
parameter-efficient fine-tuning. NCRDLLM achieves AUC-ROC values of 0.9636, 0.9662 and 0.9616 on the
miRNA-drug, lncRNA-drug, and circRNA-drug datasets, respectively.

The flow chart of NCRDLLM is as follows:

![NCRDLLM framework](NCRDLLM模型框架图.jpg)

## Directory Structure

```
├── NCRDLLM
│   ├── data                                   # Datasets for the three ncRNA types
│   │   ├── miRNA-drug
│   │   │   ├── Features_RNAFM_RNA_640D.xlsx    # RNA sequence features (640D), id column: RNA_ID
│   │   │   ├── secondary_feature_RNA.xlsx      # RNA structure features (128D), id column: RNA_ID
│   │   │   ├── semantic_miRNA_matrix_normalized.xlsx  # RNA disease semantic-similarity features (1690D), id column: RNA_ID
│   │   │   ├── Features_ChemBERTa_Drug_768D.xlsx  # Drug sequence features (768D), id column: CID
│   │   │   ├── ALLdrug-graph-features.xlsx     # Drug graph features (512D), id column: CID
│   │   │   ├── ALLdrug-ECFP-features.xlsx      # Drug ECFP fingerprints (512D), id column: CID
│   │   │   ├── semantic_Drug_matrix_normalized.xlsx  # Drug disease semantic-similarity features (1690D), id column: CID
│   │   │   ├── onehot_RNA_matrix.xlsx          # RNA-disease one-hot (used only for Jaccard negative sampling)
│   │   │   ├── onehot_Drug_matrix.xlsx         # Drug-disease one-hot (used only for Jaccard negative sampling)
│   │   │   ├── responsed_RNA-drug.xlsx         # Positive ncRNA-drug pairs, columns: RNA_ID, CID
│   │   │   └── splits                          # Auto-generated 5-fold split cache (.pkl)
│   │   ├── lncRNA-drug                         # Same file layout as miRNA-drug
│   │   └── circRNA-drug                        # Same file layout as miRNA-drug
│   ├── download_model.py                       # One-click LLaMA-3.2-3B weight downloader
│   ├── config.py                               # Global configuration
│   ├── args_parser.py                          # Command-line argument parsing
│   ├── dataset.py                              # Feature loading, negative sampling, K-fold splitting
│   ├── model.py                                # FeatureAdapter / WeightedPooling / MultimodalLLM
│   ├── utils.py                                # Metrics, seeding, early stopping
│   ├── export_utils.py                         # Feature / prediction / weight export
│   ├── visualize.py                            # ROC / PR / t-SNE plotting
│   └── train.py                                # Training code (5-fold cross-validation)
├── row                                         # Data-preprocessing scripts (for reference only; see below)
└── README.md
```

Results of each run are saved under `NCRDLLM/results/exp_<timestamp>/`, including per-fold predictions,
ROC/PR curves and the cross-validation summary.

## Installation and Requirements

NCRDLLM has been tested in a Python 3.10 environment with a CUDA-capable GPU. Using a conda virtual
environment is recommended. The recommended library versions are:

```
├── torch              2.1.1
├── transformers       4.45.2
├── peft               0.13.0
├── modelscope         1.18.0
├── scikit-learn       1.3.2
├── pandas             2.0.3
├── openpyxl           3.1.2
├── matplotlib         3.7.5
└── tqdm               4.66.1
```

### Step 1: Download Code and Data

Use the following command to download this project, or download the zip file from the "Code" section at the
top right. The multimodal feature data for all three datasets is included in the repository.

```bash
git clone https://github.com/ProZhang-Gr/NCRDLLM.git
```

### Step 2: Download Model Weights

The model reads the LLM weights locally and does not download them automatically, so the weights must be
fetched first. A one-click downloader is provided; it places the weights exactly where the code expects them
(no manual configuration needed):

```bash
cd NCRDLLM/NCRDLLM
pip install modelscope
python download_model.py
```

The weights (LLaMA-3.2-3B-Instruct, a few GB, resumable) are saved to
`NCRDLLM/NCRDLLM/llama3.1/LLM-Research/Llama-3___2-3B-Instruct/`, which is the path `train.py` looks for.

### Step 3: Run the Model

Run the main script in the virtual environment:

```bash
python train.py \
  --dataset miRNA-drug \
  --use_rna_seq --use_rna_struct --use_rna_disease \
  --use_drug_seq --use_drug_struct --use_drug_disease \
  --use_lora --lora_r 64 --lora_alpha 64
```

To switch datasets, change `--dataset` to `lncRNA-drug` or `circRNA-drug`. All results of the operation will
be saved in the `results` directory.

## Tips for a Smooth Run

A few friendly notes to help you get it running the first time:

- 🗂️ **Run from inside `NCRDLLM/NCRDLLM/`** (the folder that contains `train.py`). Both `download_model.py`
  and `train.py` should be launched from here so the paths line up automatically.
- 📋 **Just copy the run command as-is.** The `--use_*` flags and `--lora_r 64 --lora_alpha 64` are part of
  the standard setup — keeping them all is the easiest way to reproduce our results.
- 🖥️ **A CUDA GPU is needed.** Around 16 GB of VRAM is comfortable; if you hit an out-of-memory error, simply
  add `--batch_size 32` (or `16`) to the run command.
- ⏳ **The first run takes a little longer to start.** It builds and caches the cross-validation splits before
  training begins, so a quiet pause at the start is normal — just give it a moment.
- 🔍 **Seeing `未找到本地模型 / model not found`?** It just means Step 2 was skipped — run
  `python download_model.py` first and you're good to go.
- 📁 **Where are the results?** Look in `results/exp_<timestamp>/` for the predictions, ROC/PR curves and the
  cross-validation summary after the run finishes.

Everything else works out of the box — happy experimenting! 🎉

## Data Preprocessing (for reference)

The `row/` directory contains the **data-preprocessing scripts** used to build the multimodal features,
shared **for learning and reference only**. The raw and intermediate data files are very large and are
**not** included in the repository; running the final model does **not** require this directory — the
ready-to-use feature files are already provided under `NCRDLLM/data/`.

## Citation

If you use our tool and code, please cite our article and star the project to show your support, thank you!

Citation format:

Zihan Zhang, Yuchen Zhang\*, "NCRDLLM: Predicting ncRNA-Drug Response Associations via Multimodal Feature
Fusion and Large Language Models," *Journal of Chemical Information and Modeling*, 2026, online.
DOI: 10.1021/acs.jcim.5c03011. (SCI, JCR Q1, IF: 5.30)
