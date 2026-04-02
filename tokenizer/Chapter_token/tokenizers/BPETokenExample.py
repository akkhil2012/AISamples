from collections import Counter, defaultdict

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

def train_bpe(corpus, num_merges=10):
    # Initialize vocab (character-level)
    vocab = Counter()
    for word in corpus:
        vocab[" ".join(list(word)) + " </w>"] += 1

    merges = []

    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)

    return merges, vocab