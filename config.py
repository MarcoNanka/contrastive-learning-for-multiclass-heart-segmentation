"""
This code builds the command-line interface, which enables customisable model training.
"""

import argparse


parser = argparse.ArgumentParser()


args: argparse.Namespace = parser.parse_args()
