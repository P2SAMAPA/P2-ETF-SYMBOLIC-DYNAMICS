import numpy as np
from collections import Counter

def discretise_returns(series, std=None, up_thresh=0.5, down_thresh=-0.5):
    """
    Convert return series into symbolic sequence: 'U' (up), 'D' (down), 'N' (neutral).
    Thresholds are fractions of standard deviation.
    """
    if std is None:
        std = series.std()
    if std == 0:
        return ''.join(['N'] * len(series))
    symbols = []
    for r in series:
        if r > up_thresh * std:
            symbols.append('U')
        elif r < down_thresh * std:
            symbols.append('D')
        else:
            symbols.append('N')
    return ''.join(symbols)

def forbidden_words(symbol_seq, max_len=5):
    """
    Find all words (contiguous substrings) of lengths 1..max_len that never appear in the sequence.
    Returns set of forbidden words.
    """
    n = len(symbol_seq)
    alphabet = set(symbol_seq)
    all_words = set()
    for length in range(1, max_len+1):
        for i in range(n - length + 1):
            all_words.add(symbol_seq[i:i+length])
    # Total possible words of given length with alphabet size = len(alphabet)
    total_possible = set()
    # Generate all combinations? For small alphabet, brute.
    from itertools import product
    for length in range(1, max_len+1):
        for combo in product(alphabet, repeat=length):
            total_possible.add(''.join(combo))
    forbidden = total_possible - all_words
    return forbidden

def topological_entropy(symbol_seq, max_len=5):
    """
    Estimate topological entropy using the growth rate of number of distinct words.
    entropy = lim_{n->∞} log(N_n)/n, where N_n = number of distinct words of length n.
    We compute for n=1..max_len and use slope of log(N_n) vs n.
    """
    n_words = []
    for length in range(1, max_len+1):
        words = set()
        for i in range(len(symbol_seq) - length + 1):
            words.add(symbol_seq[i:i+length])
        n_words.append(len(words))
    if max_len < 2:
        return 0.0
    # Use linear regression on log(N_n) vs n
    x = np.arange(1, max_len+1)
    log_n = np.log(n_words)
    # Avoid log(0): if any n_words is 0, entropy = 0
    if np.any(np.isinf(log_n)):
        return 0.0
    # simple slope: (last - first) / (max_len-1) but more robust: linear fit
    coeff = np.polyfit(x, log_n, 1)
    return max(0.0, coeff[0])   # entropy rate

def entropy_rate(symbol_seq, max_len=5):
    """
    Estimate the entropy rate using block entropy: h = H(L+1) - H(L) where H(L) is block entropy.
    Block entropy = average uncertainty of next symbol given previous L symbols.
    """
    L = max_len
    # Count occurrences of each block of length L and L+1
    counts_L = Counter()
    counts_L1 = Counter()
    for i in range(len(symbol_seq) - L):
        block = symbol_seq[i:i+L]
        next_char = symbol_seq[i+L]
        counts_L[block] += 1
        counts_L1[block + next_char] += 1
    # Conditional entropy H(next | block) = sum p(block) * H(next|block)
    cond_ent = 0.0
    total_blocks = sum(counts_L.values())
    if total_blocks == 0:
        return 0.0
    for block, cnt in counts_L.items():
        prob_block = cnt / total_blocks
        # Distribution of next symbols given this block
        total_next = sum(v for k, v in counts_L1.items() if k.startswith(block))
        if total_next == 0:
            continue
        ent = 0.0
        for k, v in counts_L1.items():
            if k.startswith(block):
                p = v / total_next
                ent -= p * np.log(p)
        cond_ent += prob_block * ent
    return cond_ent / np.log(2)   # in bits

def predictability_score(symbol_seq, max_len=5):
    """
    Combine measures: low entropy and few forbidden words → high predictability.
    Score = (1 - entropy_rate) * (1 - forbidden_ratio) where forbidden_ratio = (#forbidden / total_possible)
    """
    # Compute entropy rate
    h = entropy_rate(symbol_seq, max_len)
    # Compute forbidden words
    forbidden = forbidden_words(symbol_seq, max_len)
    # Total possible words (simple estimate)
    alphabet = set(symbol_seq)
    total_possible = 0
    for length in range(1, max_len+1):
        total_possible += len(alphabet) ** length
    forbidden_ratio = len(forbidden) / total_possible if total_possible > 0 else 0
    # Score: low entropy and low forbidden ratio give high score
    score = (1 - min(1.0, h / np.log(len(alphabet)))) * (1 - forbidden_ratio)
    return score, h, forbidden_ratio, len(forbidden)
