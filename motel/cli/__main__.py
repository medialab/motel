#!/usr/bin/env python
# =============================================================================
# Motel CLI Endpoint
# =============================================================================
#
# CLI enpoint of the Motel library.
#
import sys
from argparse import ArgumentParser, FileType

from motel.cli.preprocess import preprocess_action

SUBPARSERS = {}


def main():
    parser = ArgumentParser(prog='motel')
    subparsers = parser.add_subparsers(help='action to execute', title='actions', dest='action')

    preprocess = subparsers.add_parser('preprocess', description='Preprocess the given corpus by splitting sentences and filtering tokens etc.')
    preprocess.add_argument(
        'column',
        help='column'
    )
    preprocess.add_argument(
        'file',
        help='csv file to cluster',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?'
    )
    preprocess.add_argument('-o',
        '--output',
        help='output file',
        type=FileType('w'),
        default=sys.stdout
    )
    preprocess.add_argument('-t',
        '--total',
        help='Total number of documents. Necessary if you want to display a finite progress indicator.',
        type=int
    )
    SUBPARSERS['preprocess'] = preprocess

    args = parser.parse_args()

    if args.action == 'help':
        target_subparser = SUBPARSERS.get(args.subcommand)

        if target_subparser is None:
            parser.print_help()
        else:
            target_subparser.print_help()

    elif args.action == 'preprocess':
        preprocess_action(args)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
