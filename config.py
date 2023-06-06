"""
This code builds the command-line interface, which enables customisable model training.
"""

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--n_of_slices_per_image', type=int, default=10,
                    help='Determines number of layers of image')

args: argparse.Namespace = parser.parse_args()
