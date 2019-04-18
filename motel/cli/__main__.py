#!/usr/bin/env python
# =============================================================================
# Motel CLI Endpoint
# =============================================================================
#
# CLI enpoint of the Motel library.
#
import sys
from argparse import ArgumentParser, FileType

SUBPARSERS = {}


def main():
    parser = ArgumentParser(prog='motel')
    subparsers = parser.add_subparsers(help='action to execute', title='actions', dest='action')

    # Preprocess subparser
    preprocess_subparser = subparsers.add_parser('preprocess', description='Preprocess the given corpus by splitting sentences and filtering tokens etc.')
    preprocess_subparser.add_argument(
        'column',
        help='column'
    )
    preprocess_subparser.add_argument(
        'file',
        help='CSV file decribing the corpus to preprocess.',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?'
    )
    preprocess_subparser.add_argument(
        '-l', '--limit',
        help='Maximum number of line to consider. Useful to perform tests.',
        type=int
    )
    preprocess_subparser.add_argument(
        '-o', '--output',
        help='Output directory.',
        default='./corpus'
    )
    preprocess_subparser.add_argument(
        '-p', '--processes',
        help='Number of processes to use. Defaults to 4.',
        type=int,
        default=4
    )
    preprocess_subparser.add_argument('-t',
        '--total',
        help='Total number of documents. Necessary if you want to display a finite progress indicator.',
        type=int
    )
    SUBPARSERS['preprocess'] = preprocess_subparser

    # Train subparser
    train_subparser = subparsers.add_parser('train', description='Train the given corpus to extract word vectors.')
    train_subparser.add_argument(
        'corpus',
        help='Corpus directory'
    )
    SUBPARSERS['train'] = train_subparser

    args = parser.parse_args()

    if args.action == 'help':
        target_subparser = SUBPARSERS.get(args.subcommand)

        if target_subparser is None:
            parser.print_help()
        else:
            target_subparser.print_help()

    elif args.action == 'preprocess':
        from motel.cli.preprocess import preprocess_action

        preprocess_action(args)

    elif args.action == 'train':
        from motel.cli.train import train_action
        
        train_action(args)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
