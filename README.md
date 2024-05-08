# Overview
Pytorch implementation of Hybrid Attention Network with FinBERT.

The original paper:
> "Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction."[[arxiv]](https://arxiv.org/abs/1712.02136)

FinBERT paper:
> "FinBERT: A Large Language Model for Extracting Information from Financial Text."[[SSRN]](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3910214)

> "Finbert: A pretrained language model for financial communications."
[[arxiv]](https://arxiv.org/abs/2006.08097)

This [[repo]](https://github.com/donghyeonk/han) is used as reference.

# Dataset
Paper:
> "Causality-Guided Multi-Memory Interaction Network for Multivariate Stock Price Movement Prediction" [[url]](https://aclanthology.org/2023.acl-long.679/)
> 
Dataset: https://github.com/BigRoddy/CMIN-Dataset

Copy it to {PROJECT_PATH}/data/

# Experiment
* pip install -r requirements.txt
* Run dataset.py
* Run model.py
  
DDP(Distributed Data Parallel) is used, so the code may not works in your environment.

# Future tasks
* Reducing dev loss vibration.