# Fuzzy Language Model

Small n-gram language model with optional fuzzy matching using Levenshtein distance.

## Project Layout

The project is now split by responsibility:

```text
|-- README.md
|-- run.py
|-- run_tests.py
|-- fuzzylm/
|   |-- __init__.py
|   |-- logic.py
|-- tests/
	|-- test_fuzzylm.py
```

Layout notes:

- `fuzzylm/logic.py`: core implementation (`NGram`, `levenshtein`, `distance_similarity`)
- `tests/test_fuzzylm.py`: unit tests
- `run.py`: quick demo runner
- `run_tests.py`: test runner

## Features

- Train unigram, bigram, trigram, ... models from tokenized text.
- Compute conditional/text log probabilities with Lidstone smoothing.
- Handle out-of-vocabulary words through fuzzy matching.

## Run Demo

```bash
python run.py
```

This trains a bigram model and scores a sample sentence.

## Run Tests

```bash
python run_tests.py
```

This executes all tests from the `tests/` folder using `unittest`.

## Import In Your Code

```python
from fuzzylm.logic import NGram

model = NGram(n=2, alpha=1.0, fuzzy=True)
```
