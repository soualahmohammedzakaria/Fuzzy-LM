# Fuzzy Language Models

Minimal implementation of an n-gram language model with fuzzy word matching based on Levenshtein distance.

## What This Project Does

This lab implements a small language model that:

- Learns unigram/bigram/trigram... counts from tokenized sentences.
- Computes conditional probabilities with Lidstone smoothing.
- Handles unknown words using fuzzy matching (Levenshtein + similarity).

## Quick Run

From the project folder:

```bash
python fuzzylms.py
```

The built-in test functions in `__main__` will execute and print assertions/results.
