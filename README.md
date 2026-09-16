# Pass-Receiver Synchronisation Audit

This repository contains the code and experimental pipeline for the study:

**“What Synchronisation Costs a Pass-Receiver Model: A Leakage Audit on a Seven-Match Public Dataset”**

The project investigates how event–tracking synchronisation affects pass-receiver prediction and whether temporal placement introduces predictive shortcuts that can inflate downstream model performance.

## Overview

The repository provides the complete experimental pipeline used in the paper, including:

- pass-receiver prediction using the canonical GRU–Transformer model;
- parameter-free ball-direction probes;
- evaluation across multiple temporal offsets;
- cross-offset evaluation of trained models;
- ball-masking experiments;
- ablation experiments;
- synthetic validation of the temporal-shape flatness criterion;
- baseline implementations;
- reproducibility and verification scripts for the reported results.

The experiments are based on the publicly available IDSSE soccer tracking and event dataset.

## Dataset

The IDSSE dataset used in this study is publicly available under a CC-BY licence from Figshare:

https://doi.org/10.6084/m9.figshare.28196177

The raw dataset is **not redistributed in this repository**. Please obtain the dataset directly from the official source and follow its licence and usage conditions.

After downloading the dataset, place or configure the data according to the paths specified in:

```text
configs/canonical.yaml
