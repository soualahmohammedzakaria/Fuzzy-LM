import math
from typing import Dict, List, Tuple


def levenshtein(w1: str, w2: str, sub: int = 2) -> int:
    """Calculates Levenshtein distance between two words."""
    if len(w1) * len(w2) == 0:
        return max(len(w1), len(w2))

    matrix = [[i for i in range(len(w2) + 1)]]
    for i in range(len(w1)):
        row = [i + 1]
        for j in range(len(w2)):
            substitution = matrix[i][j] + (0 if w1[i] == w2[j] else sub)
            min_cost = min(substitution, matrix[i][j + 1] + 1, row[j] + 1)
            row.append(min_cost)
        matrix.append(row)

    return matrix[-1][-1]


def distance_similarity(d: int, l1: int, l2: int) -> float:
    m = max(l1, l2)
    return (m - d) / m


class NGram:
    def __init__(self, alpha: float = 0.0, fuzzy: bool = False, n: int = 1):
        if n < 1:
            raise Exception(f"n={n}; n must be >= 1")
        if n > 10:
            raise Exception(f"n={n}; we limited n to 10 at most")
        if not (0.0 <= alpha <= 1.0):
            raise Exception(f"alpha={alpha} must be between 0.0 and 1.0")

        self.n = n
        self.ngrams: Dict[str, int] = {}
        self.vocab = set()
        self.alpha = alpha
        self.fuzzy = fuzzy

    def fit(self, data: List[List[str]], min_freq: int = 1):
        """Trains the n-gram model on the given data."""
        if min_freq < 1:
            raise Exception("min_freq must be 1 or plus")

        word_freq: Dict[str, int] = {}
        for sentence in data:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1

        self.vocab = set()
        has_unk = False
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.vocab.add(word)
            else:
                has_unk = True

        if has_unk:
            self.vocab.add("<unk>")

        processed = []
        for sentence in data:
            s = [word if word in self.vocab else "<unk>" for word in sentence]
            if self.n > 1:
                s = ["<s>"] * (self.n - 1) + s + ["</s>"] * (self.n - 1)
            processed.append(s)

        if self.n > 1:
            self.vocab.add("<s>")
            self.vocab.add("</s>")

        self.ngrams = {}
        for sentence in processed:
            for i in range(len(sentence) - self.n + 1):
                ngram = " ".join(sentence[i : i + self.n])
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1
                if self.n > 1:
                    context = " ".join(sentence[i : i + self.n - 1])
                    self.ngrams[context] = self.ngrams.get(context, 0) + 1

    def similar_word(self, word: str) -> Tuple[str, float]:
        """Finds the most similar word in the vocabulary to the given word."""
        best_word = ""
        best_sim = 0.0

        for v_word in self.vocab:
            d = levenshtein(word, v_word, sub=1)
            sim = distance_similarity(d, len(word), len(v_word))
            if sim > best_sim:
                best_sim = sim
                best_word = v_word

        return best_word, best_sim

    def log_cond_prob(self, current_word: str, past_words: List[str]) -> float:
        """Calculates the conditional log probability of current_word given context."""
        discount = 1.0

        if current_word not in self.vocab:
            if self.fuzzy:
                sim_word, sim = self.similar_word(current_word)
                if sim == 0:
                    current_word = "<unk>"
                else:
                    current_word = sim_word
                    discount *= sim

        if self.n == 1:
            ngram_count = self.ngrams.get(current_word, 0)
            total = sum(self.ngrams.values())
            numerator = ngram_count + self.alpha
            denominator = total + self.alpha * len(self.vocab)
        else:
            context = ["<s>"] * (self.n - 1) + list(past_words)
            context = context[-(self.n - 1) :]

            for i in range(len(context)):
                if context[i] not in self.vocab:
                    if self.fuzzy:
                        sim_word, sim = self.similar_word(context[i])
                        if sim == 0:
                            context[i] = "<unk>"
                        else:
                            context[i] = sim_word
                            discount *= sim

            ngram_key = " ".join(context + [current_word])
            context_key = " ".join(context)

            ngram_count = self.ngrams.get(ngram_key, 0)
            context_count = self.ngrams.get(context_key, 0)

            numerator = ngram_count + self.alpha
            denominator = context_count + self.alpha * len(self.vocab)

        if denominator == 0:
            return float("-inf")

        prob = (numerator / denominator) * discount

        if prob <= 0:
            return float("-inf")

        return math.log(prob)

    def log_text_prob(self, text: List[str]) -> float:
        """Calculates the log probability of a full text sequence."""
        total_log_prob = 0.0
        context: List[str] = []

        for word in text.copy():
            lp = self.log_cond_prob(word, context)
            total_log_prob += lp
            context.append(word)

        if self.n > 1:
            lp = self.log_cond_prob("</s>", context)
            total_log_prob += lp

        return total_log_prob

    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for key in data:
            self.__dict__[key] = data[key]
