"""Parser options."""

import argparse


def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(
        description='Reconstruct images from gradients.')

    # Central:
    parser.add_argument('--model', default='ConvNet',
                        type=str, help='Vision model.')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--strategy', default='conservative',
                        type=str, help='Reconstruction strategy.')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='How many images should be recovered from the given gradient.')

    # # Rec. parameters
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='Use our reconstruction method or the DLG method.')
    parser.add_argument('--max_iter', default=100, type=int,
                        help='How many iterations to run the optimizer.')
    parser.add_argument('--trials', default=1, type=int,
                        help='How many trials to run.')
    parser.add_argument('--cost_fn', default='sim', type=str,
                        help='Choice of cost function.')
    parser.add_argument('--reconstruct_label', default=False, type=bool,
                        help='Choice of cost function.')

    # # Data options
    parser.add_argument('--data_path', default='~/data', type=str,
                        help='Path to the data directory.')

    # # Advanced options
    parser.add_argument('--lr_decay', default=True, type=bool,
                        help='Decay the learning rate.')

    parser.add_argument('--idlg', default=True, type=bool,
                        help='Use the iDLG trick if batch_size = 1.')

    return parser
