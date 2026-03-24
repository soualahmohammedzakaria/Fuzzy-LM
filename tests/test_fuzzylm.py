import math
import unittest

from fuzzylm.logic import NGram, distance_similarity, levenshtein


TRAIN_TEXT = [
    "a computer can help you",
    "he wants to help you",
    "he wants a computer",
    "he can swim",
]
TRAIN_TEXT = [s.split() for s in TRAIN_TEXT]

VOCAB1 = {"a", "computer", "help", "he", "you", "to", "swim", "wants", "can"}
VOCAB2 = {
    "a",
    "computer",
    "<s>",
    "help",
    "he",
    "you",
    "</s>",
    "to",
    "wants",
    "swim",
    "can",
}

GRAMS = [
    {
        "a": 2,
        "computer": 2,
        "can": 2,
        "help": 2,
        "you": 2,
        "he": 3,
        "wants": 2,
        "to": 1,
        "swim": 1,
    },
    {
        "<s>": 4,
        "<s> a": 1,
        "a": 2,
        "a computer": 2,
        "computer": 2,
        "computer can": 1,
        "can": 2,
        "can help": 1,
        "help": 2,
        "help you": 2,
        "you": 2,
        "you </s>": 2,
        "<s> he": 3,
        "he": 3,
        "he wants": 2,
        "wants": 2,
        "wants to": 1,
        "to": 1,
        "to help": 1,
        "wants a": 1,
        "computer </s>": 1,
        "he can": 1,
        "can swim": 1,
        "swim": 1,
        "swim </s>": 1,
    },
    {
        "<s> <s>": 4,
        "<s> <s> a": 1,
        "<s> a": 1,
        "<s> a computer": 1,
        "a computer": 2,
        "a computer can": 1,
        "computer can": 1,
        "computer can help": 1,
        "can help": 1,
        "can help you": 1,
        "help you": 2,
        "help you </s>": 2,
        "you </s>": 2,
        "you </s> </s>": 2,
        "<s> <s> he": 3,
        "<s> he": 3,
        "<s> he wants": 2,
        "he wants": 2,
        "he wants to": 1,
        "wants to": 1,
        "wants to help": 1,
        "to help": 1,
        "to help you": 1,
        "he wants a": 1,
        "wants a": 1,
        "wants a computer": 1,
        "a computer </s>": 1,
        "computer </s>": 1,
        "computer </s> </s>": 1,
        "<s> he can": 1,
        "he can": 1,
        "he can swim": 1,
        "can swim": 1,
        "can swim </s>": 1,
        "swim </s>": 1,
        "swim </s> </s>": 1,
    },
    {
        "<s> <s> <s>": 4,
        "<s> <s> <s> a": 1,
        "<s> <s> a": 1,
        "<s> <s> a computer": 1,
        "<s> a computer": 1,
        "<s> a computer can": 1,
        "a computer can": 1,
        "a computer can help": 1,
        "computer can help": 1,
        "computer can help you": 1,
        "can help you": 1,
        "can help you </s>": 1,
        "help you </s>": 2,
        "help you </s> </s>": 2,
        "you </s> </s>": 2,
        "you </s> </s> </s>": 2,
        "<s> <s> <s> he": 3,
        "<s> <s> he": 3,
        "<s> <s> he wants": 2,
        "<s> he wants": 2,
        "<s> he wants to": 1,
        "he wants to": 1,
        "he wants to help": 1,
        "wants to help": 1,
        "wants to help you": 1,
        "to help you": 1,
        "to help you </s>": 1,
        "<s> he wants a": 1,
        "he wants a": 1,
        "he wants a computer": 1,
        "wants a computer": 1,
        "wants a computer </s>": 1,
        "a computer </s>": 1,
        "a computer </s> </s>": 1,
        "computer </s> </s>": 1,
        "computer </s> </s> </s>": 1,
        "<s> <s> he can": 1,
        "<s> he can": 1,
        "<s> he can swim": 1,
        "he can swim": 1,
        "he can swim </s>": 1,
        "can swim </s>": 1,
        "can swim </s> </s>": 1,
        "swim </s> </s>": 1,
        "swim </s> </s> </s>": 1,
    },
]

VOCAB3 = {"help", "computer", "you", "he", "<s>", "</s>", "wants", "can", "<unk>", "a"}
GRAMS3 = {
    "<s> a": 1,
    "<s>": 4,
    "a computer": 2,
    "a": 2,
    "computer can": 1,
    "computer": 2,
    "can help": 1,
    "can": 2,
    "help you": 2,
    "help": 2,
    "you </s>": 2,
    "you": 2,
    "<s> he": 3,
    "he wants": 2,
    "he": 3,
    "wants <unk>": 1,
    "wants": 2,
    "<unk> help": 1,
    "<unk>": 2,
    "wants a": 1,
    "computer </s>": 1,
    "he can": 1,
    "can <unk>": 1,
    "<unk> </s>": 1,
}

SIM_TEST = [
    ("plan", "can", 0.5),
    ("in", "can", 1 / 3),
    ("conputation", "computer", 0.45454545454545453),
    ("want", "wants", 0.8),
]

SENTENCE = "he can help you".split()
LOG_BIGRAM_TEST = [math.log(3 / 4), math.log(1 / 3), math.log(1 / 2), math.log(1.0)]
LOG_BIGRAM_TEST2 = [math.log(4 / 15), math.log(2 / 14), math.log(2 / 13), math.log(3 / 13)]
SENTENCE2 = "he wants to swim".split()
SENTENCE3 = "he can hope you".split()


class TestDistanceFunctions(unittest.TestCase):
    def test_levenshtein(self):
        tests = [
            ("amine", "immature", 2, 9),
            ("immature", "amine", 2, 9),
            ("", "immature", 2, 8),
            ("amine", "", 2, 5),
            ("amine", "amine", 2, 0),
            ("amine", "anine", 2, 2),
            ("amine", "anine", 1, 1),
        ]

        for w1, w2, sub, expected in tests:
            self.assertEqual(levenshtein(w1, w2, sub=sub), expected)

    def test_distance_similarity(self):
        tests = [
            (2, 5, 8, 6 / 8),
            (8, 2, 8, 0.0),
            (2, 4, 6, 4 / 6),
            (0, 5, 5, 1.0),
        ]

        for d, l1, l2, expected in tests:
            self.assertEqual(distance_similarity(d, l1, l2), expected)


class TestNGram(unittest.TestCase):
    def test_ngram_fit(self):
        for n in [1, 2, 3, 4]:
            ngram = NGram(n=n)
            ngram.fit(TRAIN_TEXT)
            self.assertEqual(ngram.vocab, VOCAB1 if n == 1 else VOCAB2)
            self.assertEqual(ngram.ngrams, GRAMS[n - 1])

        ngram = NGram(n=2)
        ngram.fit(TRAIN_TEXT, min_freq=2)
        self.assertEqual(ngram.vocab, VOCAB3)
        self.assertEqual(ngram.ngrams, GRAMS3)

    def test_similar_word(self):
        ngram = NGram()
        ngram.fit(TRAIN_TEXT)

        for word, expected_word, expected_sim in SIM_TEST:
            predicted_word, predicted_sim = ngram.similar_word(word)
            self.assertEqual(predicted_word, expected_word)
            self.assertEqual(predicted_sim, expected_sim)

    def test_log_cond_prob(self):
        ngram = NGram(n=2)
        ngram.fit(TRAIN_TEXT)

        context = []
        for i, word in enumerate(SENTENCE):
            lcp = ngram.log_cond_prob(word, context)
            context.append(word)
            self.assertEqual(round(LOG_BIGRAM_TEST[i], 10), round(lcp, 10))

        ngram.alpha = 1.0
        context = []
        for i, word in enumerate(SENTENCE):
            lcp = ngram.log_cond_prob(word, context)
            context.append(word)
            self.assertEqual(round(LOG_BIGRAM_TEST2[i], 10), round(lcp, 10))

    def test_log_text_prob(self):
        ngram = NGram(n=2)
        ngram.fit(TRAIN_TEXT)

        p = ngram.log_text_prob(SENTENCE)
        self.assertEqual(round(sum(LOG_BIGRAM_TEST), 10), round(p, 10))
        self.assertEqual(float("-inf"), ngram.log_text_prob(SENTENCE2))

        ngram.alpha = 1.0
        p = ngram.log_text_prob(SENTENCE)
        self.assertEqual(round(sum(LOG_BIGRAM_TEST2) + math.log(3 / 13), 10), round(p, 10))

        p = ngram.log_text_prob(SENTENCE2)
        self.assertEqual(round(p, 10), round(-9.010669176847115, 10))

        ngram.fuzzy = True
        p = ngram.log_text_prob(SENTENCE3)
        self.assertEqual(round(p, 10), round(-11.324304106027746, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
