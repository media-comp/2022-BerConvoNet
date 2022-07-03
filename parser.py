import argparse

parser = argparse.ArgumentParser(description='BerConvoNet for Fake News Classification')

parser.add_argument('--max-len', type=int, default=128, help='maximum length for preprocessing, 128 by default')
parser.add_argument('--batch-size', type=int, default=16, help='batch size for training, validating and testing, 16 by default')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs for training, 3 by default')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='learning rate, 1e-5 by default')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate, 0.1 by default')

args = parser.parse_args()