import argparse

import sys

sys.path.append("./")
sys.path.insert(0, "./")

import warnings

warnings.filterwarnings('ignore')

from src.bart_model.training.utils import init_logger, set_seed


def main(args):
    init_logger()
    set_seed(args)
