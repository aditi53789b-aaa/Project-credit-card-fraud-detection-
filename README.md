CTGAN for Synthetic Fraud Detection

A machine learning project that uses CTGAN (Conditional Tabular GAN) to generate synthetic financial transaction data for fraud detection research while preserving privacy.

 Project Overview

Real fraud datasets are highly imbalanced and often contain sensitive customer information. This project demonstrates how CTGAN can learn the distribution of tabular transaction data and generate realistic synthetic samples that can be used for:

Fraud detection model training

Data augmentation

Research and experimentation

Privacy-preserving analytics

 Features

Loads and preprocesses transaction data
Trains a CTGAN model on tabular data
Generates synthetic fraud and non-fraud transactions
Compares real vs synthetic data distributions
Preserves privacy by avoiding exposure of real customer records

 Tech Stack

Python

Pandas

NumPy

Matplotlib

Scikit-learn

CTGAN / SDV


 Output

The notebook generates:

Synthetic transaction records

Statistical comparison of real and synthetic data

Visualizations of feature distributions

Samples of generated fraud transactions
<img width="740" height="140" alt="image" src="https://github.com/user-attachments/assets/19c51958-217e-492c-b961-989c32022bcb" />
