# =============================================================================
# Motel Preprocess CLI Action
# =============================================================================
#
# CLI action ingesting a corpus and preprocessing documents to generate a
# corpus so we can apply word2vec on it later. Typically, here we deal with
# sentence segmentation, tokenization, lemmatization etc.
#
import os
import csv
import spacy
from collections import Counter
from multiprocessing import Pool
from os.path import join
from tqdm import tqdm
from gensim.utils import deaccent
from motel.cli.utils import custom_reader

OUTPUT_HEADERS = ['id', 'tokens']
VOCAB_HEADERS = ['word', 'count']

NLP = None

# Helpers
def filter_token(token):
    if token.is_stop or token.is_digit or token.is_punct or token.is_space:
        return False

    if token.like_url or token.like_email or token.like_num:
        return False

    return True


def process_token(token):
    return deaccent(token.lemma_.lower())


# Worker-related
def initializer():
    global NLP

    # NOTE: handle language later!
    NLP = spacy.load('fr', disable=('ner', 'tagger', 'textcat'))
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))


def worker(payload):
    i, text, line = payload

    vocab = Counter()

    doc = NLP(text)

    sentences = []

    for sentence in doc.sents:

        # Filtering some tokens
        sentence = [token for token in sentence if filter_token(token)]

        if len(sentence) == 0:
            continue

        # Counting tokens
        tokens = [process_token(token) for token in sentence]

        for token in tokens:
            vocab[token] += 1

        sentences.append(tokens)

    return i, line, sentences, vocab


def preprocess_action(namespace):
    headers, pos, reader = custom_reader(namespace.file, namespace.column)

    os.makedirs(namespace.output, exist_ok=True)

    sf = open(join(namespace.output, 'sentences.csv'), 'w')
    writer = csv.writer(sf)
    writer.writerow(OUTPUT_HEADERS)

    loading_bar = tqdm(
        desc='Preprocessing documents',
        total=namespace.total,
        dynamic_ncols=True,
        unit=' docs'
    )

    full_vocab = Counter()

    pool = Pool(namespace.processes, initializer=initializer)

    generator = ((i, line[pos], line) for i, line in enumerate(reader))

    for i, line, sentences, vocab in pool.imap_unordered(worker, generator):
        loading_bar.update()

        full_vocab += vocab

        for tokens in sentences:
            writer.writerow([i, '§'.join(tokens)])

    sf.close()

    # Outputting vocabulary
    print('Writing vocabulary...')
    with open(join(namespace.output, 'vocab.csv'), 'w') as vf:
        writer = csv.writer()
        writer.writerow(VOCAB_HEADERS)

        for item, count in full_vocab.most_common():
            writer.writerow([item, count])
