import torch
import logging


def setup_reproducibility(seed=0, deterministic=True, benchmark=False):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    logging.info('torch.backends.cudnn.deterministic = %s' % str(deterministic))
    logging.info('torch.backends.cudnn.benchmark = %s' % str(benchmark))


def setup_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(filename=path)
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device
