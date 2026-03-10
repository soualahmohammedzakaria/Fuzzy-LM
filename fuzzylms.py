#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QUESTIONS/ANSWERS

Short version
----------------------------------------------------------
Q1: Distance vs. Similarity
1.1. Why we are taking the max between the two lengths and not the average?
1.2. Our similarity functions only if insertion=deletion=substitution=1.
     How can we adapt it to a parametric Levenshtein distance?
----------------------------------------------------------
A1:
1.1. Using the maximum of the two lengths guarantees that the similarity score
     stays in the range [0, 1]. The maximum possible edit distance between two
     words (with ins=del=sub=1) equals max(l1, l2), since in the worst case we
     delete the entire longer word and rebuild it. If we used the average
     (l1+l2)/2, the distance could exceed that average (e.g. l1=1, l2=10,
     d=10 > avg=5.5), producing a negative similarity. The max therefore
     provides a correct and conservative normalization.
1.2. When the operation costs differ, the maximum possible distance is no longer
     max(l1, l2). It becomes the cost of deleting all of w1 and inserting all
     of w2: max_d = l1 * cost_del + l2 * cost_ins. The adapted similarity is:
     sim = (max_d - d) / max_d, where d is the parametric Levenshtein distance.
     This ensures the result remains in [0, 1] for any combination of costs.

----------------------------------------------------------
Q2: Fuzzy search
2.1. Can fuzzy search fix out-of-vocabulary problem? Explain.
2.2. Will it always do so? Explain.
----------------------------------------------------------
A2: 
2.1. Yes, partially. When an OOV word is a misspelling or a morphological
     variant of a known vocabulary word (for example, "compter" -> "computer"),
     fuzzy search finds the closest match via Levenshtein distance and uses it
     as a replacement, allowing the model to assign a non-zero probability to
     the sequence instead of failing.
2.2. No. Fuzzy search can fail when:
     (a) the OOV word is genuinely new and unrelated to any vocabulary word, so
     the closest match has a different meaning,
     (b) multiple vocabulary words have similar distances, and the wrong one is
     selected,
     (c) the similarity score is zero, meaning no reasonable match exists,
     (d) short words are easily confused with unrelated short words (for example,
     "in" matched to "can").

"""


import math
import json
import random
from typing import List, Dict, Tuple

#=============================================================================
#                             Functions
#=============================================================================

def levenshtein(w1:str, w2:str, sub:int=2) -> int:
    """Calculates Levenshtein distance between two words.
    The function's words are interchangeable; i.e. levenshtein(w1, w2) = levenshtein(w2, w1)

    Args:
        w1 (str): First word.
        w2 (str): Second word.
        sub (int, optional): Substitution's cost. Defaults to 2.

    Returns:
        int: distance
    """

    if len(w1) * len(w2) == 0:
        return max([len(w1), len(w2)])

    D = []
    D.append([i for i in range(len(w2) + 1)])
    for i in range(len(w1)):
        l = [i+1]
        for j in range(len(w2)):
            s = D[i][j] + (0 if w1[i] == w2[j] else sub)
            m = min([s, D[i][j+1] + 1, l[j] + 1])
            l.append(m)
        D.append(l)

    return D[-1][-1]


def distance_similarity(d: int, l1: int, l2: int) -> float:
    m = max(l1, l2)
    return (m-d)/m

#=============================================================================
#                             To implement
#=============================================================================

class NGram:

    def __init__(self, alpha: float = 0.0, fuzzy: bool = False, n: int = 1):
        if n < 1:
            raise Exception(f"n={n}; n must be >= 1")
        if n > 10:
            raise Exception(f"n={n}; we limited n to 10 at most")
        
        if not (0.0 <= alpha <= 1.):
            raise Exception(f"alpha={alpha} must be between 0.0 and 1.0")
        
        self.n = n
        self.ngrams: Dict[str, int] = {}
        self.vocab = set()
        self.alpha = alpha
        self.fuzzy = fuzzy


    # TODO Complete
    def fit(self, data:List[List[str]], min_freq: int = 1):
        """
        Trains the n-gram model on the given data.

        Args:
            data (List[List[str]]): A list of sentences, where each sentence is itself a list of words (tokens).
                Example: [['the', 'cat', 'sat'], ['a', 'dog', 'ran']]
            min_freq (int): Minimum frequency threshold for n-grams to be included.
        Returns:
            None: This method updates the model in-place and does not return anything.
        """
        if min_freq < 1:
            raise Exception("min_freq must be 1 or plus")

        # Count word frequencies
        word_freq = {}
        for sentence in data:
            for word in sentence:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Build vocabulary based on min_freq
        self.vocab = set()
        has_unk = False
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.vocab.add(word)
            else:
                has_unk = True

        if has_unk:
            self.vocab.add('<unk>')

        # Process sentences: replace rare words with <unk> and add padding
        processed = []
        for sentence in data:
            s = [word if word in self.vocab else '<unk>' for word in sentence]
            if self.n > 1:
                s = ['<s>'] * (self.n - 1) + s + ['</s>'] * (self.n - 1)
            processed.append(s)

        if self.n > 1:
            self.vocab.add('<s>')
            self.vocab.add('</s>')

        # Count n-grams and (n-1)-grams
        self.ngrams = {}
        for sentence in processed:
            for i in range(len(sentence) - self.n + 1):
                ngram = ' '.join(sentence[i:i + self.n])
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1
                if self.n > 1:
                    context = ' '.join(sentence[i:i + self.n - 1])
                    self.ngrams[context] = self.ngrams.get(context, 0) + 1


    # TODO Complete
    def similar_word(self, word: str) -> Tuple[str, float]:
        """
            Finds the most similar word in the vocabulary to the given word.
            
            Args:
                word (str): The word to find a similar match for.
                
            Returns:
                Tuple[str, float]: The most similar word from vocab and its similarity score.
            """
        
        # Note: in log_cond_prob, when sim == 0 the code ignores the returned word and falls back to <unk>
        # So, in short, no we are not wrong when we return "" if all similarities are 0
        best_word = ""
        best_sim = 0.0

        for v_word in self.vocab:
            d = levenshtein(word, v_word, sub=1)
            sim = distance_similarity(d, len(word), len(v_word))
            if sim > best_sim:
                best_sim = sim
                best_word = v_word

        return best_word, best_sim


    # TODO Complete
    def log_cond_prob(self, current_word: str, past_words: List[str]) -> float:
        """
        Calculates the conditional log probability of the current_word given the past_words context.

        Args:
            current_word (str): The word for which the probability is to be computed.
            past_words (List[str]): The list of previous words (context).

        Returns:
            float: The conditional log probability of current_word given past_words.
        """
        discount = 1.0

        # Fuzzy matching for current_word
        if current_word not in self.vocab:
            if self.fuzzy:
                sim_word, sim = self.similar_word(current_word)
                if sim == 0:
                    current_word = '<unk>'
                else:
                    current_word = sim_word
                    discount *= sim

        if self.n == 1:
            ngram_count = self.ngrams.get(current_word, 0)
            total = sum(self.ngrams.values())
            numerator = ngram_count + self.alpha
            denominator = total + self.alpha * len(self.vocab)
        else:
            context = ['<s>'] * (self.n - 1) + list(past_words)
            context = context[-(self.n - 1):]

            # Fuzzy matching for context words
            for i in range(len(context)):
                if context[i] not in self.vocab:
                    if self.fuzzy:
                        sim_word, sim = self.similar_word(context[i])
                        if sim == 0:
                            context[i] = '<unk>'
                        else:
                            context[i] = sim_word
                            discount *= sim

            ngram_key = ' '.join(context + [current_word])
            context_key = ' '.join(context)

            ngram_count = self.ngrams.get(ngram_key, 0)
            context_count = self.ngrams.get(context_key, 0)

            numerator = ngram_count + self.alpha
            denominator = context_count + self.alpha * len(self.vocab)

        if denominator == 0:
            return float('-inf')

        prob = (numerator / denominator) * discount

        if prob <= 0:
            return float('-inf')

        return math.log(prob)

    # TODO Complete
    def log_text_prob(self, text: List[str]) -> float:
        """
        Calculates the log probability of a given text sequence.
        Args:
            text (List[str]): A list of words/tokens representing the text sequence.
        Returns:
            float: The log probability of the text sequence.
        """
        text = text.copy()
        total_log_prob = 0.0
        context = []
        for word in text:
            lp = self.log_cond_prob(word, context)
            total_log_prob += lp
            context.append(word)

        if self.n > 1:
            lp = self.log_cond_prob('</s>', context)
            total_log_prob += lp

        return total_log_prob

    def export_json(self):
        return self.__dict__.copy()

    def import_json(self, data):
        for cle in data:
            self.__dict__[cle] = data[cle]

    
#=============================================================================
#                             TESTS
#=============================================================================

def _levenshtein_test():
    tests = [
        ('amine', 'immature', 2, 9),
        ('immature', 'amine', 2, 9),
        ('', 'immature', 2, 8),
        ('amine', '', 2, 5),
        ('amine', 'amine', 2, 0),
        ('amine', 'anine', 2, 2),
        ('amine', 'anine', 1, 1),
    ]
    
    for test in tests:
        d = levenshtein(test[0], test[1], sub=test[2])
        print('-----------------------------------')
        print(f'levenshtein({test[0]}, {test[1]})={d}; must be {test[3]}')

def _distance_similarity_test():
    tests = [
        (2, 5, 8, 6/8),
        (8, 2, 8, 0.),
        (2, 4, 6, 4/6),
        (0, 5, 5, 1.),
    ]
    
    for test in tests:
        sim = distance_similarity(test[0], test[1], test[2])
        print('-----------------------------------')
        print(f'sim={sim}; must be {test[3]}')
        assert sim == test[3] # will raise an exception if false

train_text = [
    "a computer can help you",
    "he wants to help you",
    "he wants a computer",
    "he can swim"
]

train_text = [s.split() for s in train_text]

vocab1 = {'a', 'computer', 'help', 'he', 'you', 'to', 'swim', 'wants', 'can'}
vocab2 = {'a', 'computer', '<s>', 'help', 'he', 'you', '</s>', 'to', 'wants', 'swim', 'can'}

grams = [
    {'a': 2, 'computer': 2, 'can': 2, 'help': 2, 'you': 2, 'he': 3, 'wants': 2, 'to': 1, 'swim': 1},
    {'<s>': 4, '<s> a': 1, 'a': 2, 'a computer': 2, 'computer': 2, 'computer can': 1, 'can': 2, 'can help': 1, 
     'help': 2, 'help you': 2, 'you': 2, 'you </s>': 2, '<s> he': 3, 'he': 3, 'he wants': 2, 'wants': 2, 'wants to': 1, 
     'to': 1, 'to help': 1, 'wants a': 1, 'computer </s>': 1, 'he can': 1, 'can swim': 1, 'swim': 1, 'swim </s>': 1},
    {'<s> <s>': 4, '<s> <s> a': 1, '<s> a': 1, '<s> a computer': 1, 'a computer': 2, 'a computer can': 1, 'computer can': 1, 
     'computer can help': 1, 'can help': 1, 'can help you': 1, 'help you': 2, 'help you </s>': 2, 'you </s>': 2, 
     'you </s> </s>': 2, '<s> <s> he': 3, '<s> he': 3, '<s> he wants': 2, 'he wants': 2, 'he wants to': 1, 'wants to': 1, 
     'wants to help': 1, 'to help': 1, 'to help you': 1, 'he wants a': 1, 'wants a': 1, 'wants a computer': 1, 
     'a computer </s>': 1, 'computer </s>': 1, 'computer </s> </s>': 1, '<s> he can': 1, 'he can': 1, 'he can swim': 1, 
     'can swim': 1, 'can swim </s>': 1, 'swim </s>': 1, 'swim </s> </s>': 1},
    {'<s> <s> <s>': 4, '<s> <s> <s> a': 1, '<s> <s> a': 1, '<s> <s> a computer': 1, '<s> a computer': 1, 
     '<s> a computer can': 1, 'a computer can': 1, 'a computer can help': 1, 'computer can help': 1, 
     'computer can help you': 1, 'can help you': 1, 'can help you </s>': 1, 'help you </s>': 2, 
     'help you </s> </s>': 2, 'you </s> </s>': 2, 'you </s> </s> </s>': 2, '<s> <s> <s> he': 3, 
     '<s> <s> he': 3, '<s> <s> he wants': 2, '<s> he wants': 2, '<s> he wants to': 1, 'he wants to': 1, 
     'he wants to help': 1, 'wants to help': 1, 'wants to help you': 1, 'to help you': 1, 'to help you </s>': 1, 
     '<s> he wants a': 1, 'he wants a': 1, 'he wants a computer': 1, 'wants a computer': 1, 'wants a computer </s>': 1, 
     'a computer </s>': 1, 'a computer </s> </s>': 1, 'computer </s> </s>': 1, 'computer </s> </s> </s>': 1, 
     '<s> <s> he can': 1, '<s> he can': 1, '<s> he can swim': 1, 'he can swim': 1, 'he can swim </s>': 1, 'can swim </s>': 1, 
     'can swim </s> </s>': 1, 'swim </s> </s>': 1, 'swim </s> </s> </s>': 1}

]

vocab3 = {'help', 'computer', 'you', 'he', '<s>', '</s>', 'wants', 'can', '<unk>', 'a'}
grams3 = {'<s> a': 1, '<s>': 4, 'a computer': 2, 'a': 2, 'computer can': 1, 'computer': 2, 
          'can help': 1, 'can': 2, 'help you': 2, 'help': 2, 'you </s>': 2, 'you': 2, '<s> he': 3, 'he wants': 2, 
          'he': 3, 'wants <unk>': 1, 'wants': 2, '<unk> help': 1, '<unk>': 2, 'wants a': 1, 'computer </s>': 1, 
          'he can': 1, 'can <unk>': 1, '<unk> </s>': 1}

def _ngram_fit_test():
    for n in [1, 2, 3, 4]:
        ngram = NGram(n=n)
        ngram.fit(train_text)
        print(f"n={n}")
        assert ngram.vocab == (vocab1 if n==1 else vocab2)
        print("vocabulary asserted")
        assert ngram.ngrams == grams[n-1]
        print("grams asserted")
        print("------------")

    # ----------------
    ngram = NGram(n=2)
    ngram.fit(train_text, min_freq=2)
    print(f"n=2 with min_freq=2")
    assert ngram.vocab == vocab3
    print("vocabulary asserted")
    assert ngram.ngrams == grams3
    print("grams asserted")
    print("------------")
    

sim_test = [
    ("plan", "can", 0.5),
    ("in", "can", 0.3333333333333333),
    ("conputation", "computer", 0.45454545454545453),
    ("want", "wants", 0.8)
]

def _ngram_similar_word_test():
    ngram = NGram()
    ngram.fit(train_text)
    for word, sword, sim in sim_test:
        sword_pred, sim_pred = ngram.similar_word(word)
        assert sword_pred == sword
        assert sim_pred == sim
        print(f"\"{word}\" asserted")


sentence = "he can help you".split()
log_bigram_tst = [
    math.log(3/4), math.log(1/3), math.log(1/2), math.log(2/2)
]

log_bigram_tst2 = [
    math.log(4/15), math.log(2/14), math.log(2/13), math.log(3/13)
]

sentence2 = "he wants to swim".split()

sentence3 = "he can hope you".split()

def _ngram_log_cond_prob_test():
    ngram = NGram(n=2)
    ngram.fit(train_text)

    context = []
    for i, word in enumerate(sentence):
        lcp = ngram.log_cond_prob(word, context)
        context += [word]
        assert round(log_bigram_tst[i], 10) == round(lcp, 10)
    print("Bigrams asserted")

    ngram.alpha = 1.
    context = []
    for i, word in enumerate(sentence):
        lcp = ngram.log_cond_prob(word, context)
        context += [word]
        assert round(log_bigram_tst2[i], 10) == round(lcp, 10)
    print("Bigrams laplace asserted")
    

def _ngram_log_text_prob_test():
    ngram = NGram(n=2)
    ngram.fit(train_text)

    p = ngram.log_text_prob(sentence)
    assert round(sum(log_bigram_tst), 10) == round(p, 10)
    # print(ngram.log_text_prob(sentence2))
    assert float('-inf') == ngram.log_text_prob(sentence2)
    print("Bigrams no smoothing no fuzzy asserted")

    ngram.alpha = 1.
    p = ngram.log_text_prob(sentence)
    assert round(sum(log_bigram_tst2) + math.log(3/13), 10) == round(p, 10)
    p = ngram.log_text_prob(sentence2)
    assert round(p, 10) == round(-9.010669176847115, 10)
    print("Bigrams laplace asserted")

    ngram.fuzzy = True
    p = ngram.log_text_prob(sentence3)
    assert round(p, 10) == round(-11.324304106027746, 10)
    print("Bigrams smoothing no fuzzy asserted")





if __name__ == '__main__':
    _levenshtein_test()
    _distance_similarity_test() 

    # Start here; the two past functions are given
    _ngram_fit_test()
    _ngram_similar_word_test()
    _ngram_log_cond_prob_test()
    _ngram_log_text_prob_test()

    
