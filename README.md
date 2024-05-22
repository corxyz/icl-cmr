# Linking In-context Learning in Transformers to Human Episodic Memory

This repository is the official implementation of [Linking In-context Learning in Transformers to Human Episodic Memory](https://arxiv.org/abs/2030.12345). 

![Tux, the Linux mascot](/figs/comparison.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Quick Start

For a quick demonstration of how attention heads in Transformer models share key features with human episodic memory, start from our Google Colab [Demo Notebook](demo.ipynb). We used the GPT2-small model from [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to show

- CRP (Conditional Response Probability) analysis of attention heads
- Computing metrics for induction heads (induction-head matching score, copying score)
- Comparing induction heads to episodic retrieval mediated by a temporal context

## Experiments

To replicate the experiments, first obtain (1) the attention & head scores of pre-trained LLMs by running the Demo Notebook (some of them may take a couple of hours - check the paper appendix for details) and save them under ```saved_scores/<model_name>/``` (e.g., ```saved_scores/gpt2-small/```), as well as (2) the CRPs produced by CMR by running

```
python src/est_cmr_crp.py
```

The command above can take a few days to complete; we have provided the results under ```saved_crps/```.

#### 1. Probing Individual Attention Heads

To examine individual attention heads in a model (e.g., Figure 5 and 6a in the paper), run

```
python src/fit_attn_score.py --save_dir <directory_to_save_results> --all_head_scores <path_to_saved_scores>
```

#### 2. Comparing models of different sizes

To examine where in the model do attention heads most likely exhibit a human-like asymmetric contiguity bias (e.g., Figure 6b), run

```
python src/attn_over_size.py --model_size <model_size_1> <model_size_2> ...
```

For example, ```python src/attn_over_size.py --model_size 70m 160m 410m 1b``` compares the smaller LLMs (Pythia-70m-deduped-v0, Pythia-160m-deduped-v0, Pythia-410m-deduped-v0, Pythia-1b-deduped-v0).

#### 3. Comparing the same model across different training steps

To examine how each model evolves over training in terms of memory-related features (e.g., Figure 7b, Supplementary Figure S2), run

```
python src/attn_over_time.py --model_name <model_name>
```

For example, ```python src/attn_over_time.py --model_name pythia-70m-deduped-v0``` analyzes the Pythia-70m-deduped-v0 model at different training checkpoints by fitting each attention head to the CMR model.

#### 4. Comparing behavior of top induction/CMR heads

To examine the memory-related features of top heads (e.g., Figure 7c-f, Supplementary Figure S3), run

```
python src/select_fit.py --model_name <model_name_1> <model_name_2> ... --n_top <top_set_size_1> <top_set_size_2> ... 
```

For example, ```python src/select_fit.py --n_top 20 50 100 200``` finds the top 20/50/100/200 CMR heads and induction heads.

## Pre-trained Models

We used the pre-trained models from [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to generate the results in our paper, including

- GPT2-small
- Pythia-70m-deduped-v0
- Pythia-160m-deduped-v0
- Pythia-410m-deduped-v0
- Pythia-1b-deduped-v0
- Pythia-1.4b-deduped-v0
- Pythia-2.8b-deduped-v0
- Pythia-6.9b-deduped-v0
- Pythia-12b-deduped-v0
