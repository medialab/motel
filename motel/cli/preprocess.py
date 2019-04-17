# =============================================================================
# Motel Preprocess CLI Action
# =============================================================================
#
# CLI action ingesting a corpus and preprocessing documents to generate a
# corpus so we can apply word2vec on it later. Typically, here we deal with
# sentence segmentation, tokenization, lemmatization etc.
#
import csv
import spacy
from tqdm import tqdm
from motel.cli.utils import custom_reader

OUTPUT_HEADERS = ['id', 'tokens']


def filter_token(token):
    if token.is_stop or token.is_digit or token.is_punct or token.is_space:
        return False

    if token.like_url or token.like_email or token.like_num:
        return False

    return True


def process_token(token):
    return token.lemma_.lower()


def preprocess_action(namespace):

    # NOTE: handle language later!
    nlp = spacy.load('fr', disable=('ner', 'tagger', 'textcat', 'parser'))
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    headers, pos, reader = custom_reader(namespace.file, namespace.column)

    writer = csv.writer(namespace.output)
    writer.writerow(OUTPUT_HEADERS)

    loading_bar = tqdm(
        desc='Preprocessing documents',
        total=namespace.total,
        dynamic_ncols=True,
        unit=' docs'
    )

    for i, line in enumerate(reader):
        doc = nlp(line[pos])
        loading_bar.update()

        for sentence in doc.sents:

            # Filtering some tokens
            sentence = [token for token in sentence if filter_token(token)]

            if len(sentence) == 0:
                continue

            # Outputting a sentence
            writer.writerow([i, '§'.join(process_token(token) for token in sentence)])
