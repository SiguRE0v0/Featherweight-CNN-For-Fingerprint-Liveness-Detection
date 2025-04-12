from Src import train, predict
from Utils.config import load_config
import argparse
import logging


def main():
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser.add_argument('--config', type=str, default='config.yml', help='path of yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    logging.info('Config loaded')

    mode = config.mode
    if mode == 'train':
        train(config)
    elif mode == 'predict':
        predict(config)
    else:
        logging.error('mode configration should be train or predict')


if __name__ == '__main__':
    main()
