# =============================================================================
# Motel Train CLI Action
# =============================================================================
#
# CLI action training word vectors from a corpus of preprocessed sentences.
#
import csv
import json
from os.path import join
from gensim.models import Word2Vec
from collections import Counter


def train_action(namespace):

    model = Word2Vec()

    # Reading stats
    with open(join(namespace.corpus, 'stats.json')) as stf:
        stats = json.load(stf)

    vocab_freqs = Counter()

    # Building vocab
    print('Reading vocabulary...')
    with open(join(namespace.corpus, 'vocab.csv')) as vf:
        reader = csv.reader(vf)
        next(reader)

        for line in reader:
            word = line[0]
            count = int(line[1])

            if count < 5:
                break

            vocab_freqs[word] = count

    print('Building vocabulary (%i items)...' % len(vocab_freqs))
    model.build_vocab_from_freq(vocab_freqs)

    print('Training vectors...')
    with open(join(namespace.corpus, 'sentences.csv')) as sf:
        reader = csv.reader(sf)
        next(reader)

        sentences = (line[1].split('§') for line in reader)

        model.train(sentences, total_examples=stats['sentences'], epochs=5)
