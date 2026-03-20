# LLMMHST

![](https://img.shields.io/badge/Under-review-blue)
![](https://img.shields.io/badge/python-3.8.19-orange)
![](https://img.shields.io/badge/pytorch-2.2.0-orange)
![](https://img.shields.io/badge/pytorchlightning-2.3.0-orange)

Implementation of the paper “Multi-Scale Spatial-Temporal Graph Learning for Highway Overload Traffic Prediction via LLM-Guided Augmentation.”

## Overview
LLMMHST is a tailored framework for highway traffic flow prediction under overload scenarios using LLM-Guided Data Augmentation. Specifically, we employ a frozen pretrained LLM on normal-scenario spatial-temporal traffic records as a prompt-conditioned proposal generator. It outputs latent semantic prior representations describing high-level overload traffic patterns. A temporal translator is used for these proposals to enforce temporal coherence, while a frequency-domain mapper distills spatial-temporal priors from limited real overload samples to refine the proposals via cross-attention. The augmented dataset improves model robustness under data-scarce overload scenarios. We then construct a heterogeneous traffic graph to depict various transfer interactions on highway networks. We develop a multi-scale weaving Transformer network to adapt to irregular traffic patterns. A coupled heterogeneous graph attention network performed on the traffic graph is delivered to learn the complex traffic behaviors. Both types of networks learn alternately to form a multi-scale heterogeneous spatial-temporal module as the primary learner.

## Usage
You need to install some necessary packages from requirements.txt.
```bash
pip install -r requirements.txt
```

## Quick start
You can run the following command to train the HST-WAVE.
```bash
python train.py
```
