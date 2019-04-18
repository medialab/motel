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
from os.path import join
from tqdm import tqdm
from gensim.utils import deaccent
from motel.cli.utils import custom_reader

OUTPUT_HEADERS = ['id', 'tokens']


def filter_token(token):
    if token.is_stop or token.is_digit or token.is_punct or token.is_space:
        return False

    if token.like_url or token.like_email or token.like_num:
        return False

    return True


def process_token(token):
    return deaccent(token.lemma_.lower())


def preprocess_action(namespace):

    # NOTE: handle language later!
    nlp = spacy.load('fr', disable=('ner', 'tagger', 'textcat'))
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    headers, pos, reader = custom_reader(namespace.file, namespace.column)

    os.makedirs(namespace.output, exist_ok=True)

    writer = csv.writer(join(namespace.output, 'sentences.csv'))
    writer.writerow(OUTPUT_HEADERS)

    loading_bar = tqdm(
        desc='Preprocessing documents',
        total=namespace.total,
        dynamic_ncols=True,
        unit=' docs'
    )

    vocab = Counter()

    for i, line in enumerate(reader):
        doc = nlp(line[pos])
        loading_bar.update()

        for sentence in doc.sents:

            # Filtering some tokens
            sentence = [token for token in sentence if filter_token(token)]

            if len(sentence) == 0:
                continue

            # Counting tokens
            tokens = [process_token(token) for token in sentence]

            for token in tokens:
                vocab[token] += 1

            # Outputting a sentence
            writer.writerow([i, '§'.join(tokens)])

    # Outputting vocabulary
    print('Writing vocabulary...')
    with open(join(namespace.output, 'vocab.csv'), 'w') as vf:
        writer = csv.writer()
        writer.writerow(['word', 'count'])

        for item, count in vocab.most_common():
            writer.writerow([item, count])
