# SMS Spam or Ham Classification (Beginner)

A beginner-friendly NLP project to classify SMS messages as **spam** or **ham** using simple machine learning techniques.

## Overview

This repository provides a walk-through SMS classification pipeline using the SMS Spam Collection Dataset—a collection of 5,572 labeled messages (Ham: 4,825, Spam: 747) :contentReference[oaicite:1]{index=1}.

You’ll learn:
- How to preprocess and clean text data
- Basic feature engineering for SMS content
- Training and evaluating a machine learning classifier for spam detection

## Dataset

- **Source**: Kaggle notebook by dejavu23, originally using the SMS Spam Collection Dataset from UCI :contentReference[oaicite:2]{index=2}.
- **Size**: 5,572 SMS messages
  - 4,825 labeled as *ham* (legitimate)
  - 747 labeled as *spam*

## Project Structure

├── .gitattributes
├── .gitignore
│── LICENSE
├── spam_ham_main.py # Kaggle notebook adapted to local use
├── README.md
└── requirements.txt
└── spam.csv
