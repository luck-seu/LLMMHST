# LLMMHST

![](https://img.shields.io/badge/Under-review-blue)
![](https://img.shields.io/badge/python-3.8.19-orange)
![](https://img.shields.io/badge/pytorch-2.2.0-orange)
![](https://img.shields.io/badge/pytorchlightning-2.3.0-orange)

Implementation of the paper “Traffic Flow Prediction on Overloaded Highways Using LLM-Guided Data Augmentation”

## Overview
LLMMHST is a tailored framework for highway traffic flow prediction under overload scenarios using LLM-Guided Data Augmentation. Specifically, we first fine-tune a pre-trained LLM with normal- and overload-scenario spatial-temporal traffic records to embed priors into its latent representation space. The adapted LLM is subsequently employed to generate overload traffic data by conditioning on normal-scenario textual prompts, producing samples with realistic spatial-temporal patterns aligned with source distributions. These generated data augment the training set, improving model robustness under data-scarce overload scenarios. We then construct a heterogeneous traffic graph to depict various transfer interactions on highway networks. We develop a multi-scale weaving Transformer network to adapt to irregular traffic patterns. A coupled heterogeneous graph attention network performed on the traffic graph is delivered to learn the complex traffic behaviors. Both types of networks learn alternately to form a multi-scale heterogeneous spatial-temporal module as the primary learner.

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
