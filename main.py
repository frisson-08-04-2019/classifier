import torch
import logging
import datetime
import os
import argparse
import numpy as np

from classifier.classifier import Classifier
from classifier.utils import setup_reproducibility, setup_logger, get_device

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/data')
    parser.add_argument('--output_path', default='output')
    parser.add_argument('--experiment_name', default=datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=0.0008, type=int)
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--units', default=100, type=int)

    parser.add_argument('--seed', default=np.random.randint(0, 2 ** 31 - 1), type=int)
    parser.add_argument('--load_net_path', default=None)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_ds', default='eval')
    parser.add_argument('--misclassified', action='store_true')
    parser.add_argument('--eval_single', action='store_true')
    parser.add_argument('--eval_single_img', default=None)

    args = parser.parse_args()
    results_path = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(results_path)

    setup_logger(os.path.join(results_path, 'experiment.log'))
    setup_reproducibility(args.seed)
    device = get_device()
    logging.info('Using device: %s' % device)

    logging.info('Arguments:')
    for arg, val in vars(args).items():
        logging.info('%s: %s' % (arg, val))

    classifier = Classifier(data_path=args.data_path,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum,
                            epochs=args.epochs,
                            dropout=args.dropout,
                            batch_size=args.batch_size,
                            results_path=results_path,
                            units=args.units,
                            load_net_path=args.load_net_path)

    if args.train:
        classifier.train()

    if args.test:
        classifier.test(ds='test')

    if args.eval:
        classifier.eval(ds=args.eval_ds)

    if args.misclassified:
        classifier.get_misclassified(ds=args.misclassified_ds)

    if args.eval_single:
        print(classifier.eval_single_img(path=args.eval_single_img))
