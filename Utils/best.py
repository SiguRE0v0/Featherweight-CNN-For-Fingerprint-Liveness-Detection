import sys
sys.path.append('/root/Fingerprint_liveness_detection/')

import os
import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import build_dataset
from Model import FLDNet
from datetime import datetime
from evaluate import Metrics
import argparse
from config import load_config
import shutil
from tqdm import tqdm


def find_best_model(config):
    device = config.device
    logging.info(f'Using device {device}')
    if config.load_model is None:
        raise ValueError('No model loaded for continue training')
    in_channels = config.input_channels
    out_classes = config.out_classes
    model = FLDNet(in_channels=in_channels, out_classes=out_classes, enable_se=config.enable_SE,
                   spp=config.enable_SPP, invert=config.invert).to(device)
    transform = transforms.Compose([transforms.RandomCrop((112, 112))]) if config.enable_crop else None
    test_set, _ = build_dataset(img_dir=config.test_data_path, val_percent=0,
                                img_size=config.image_size, transform=transform)
    rounds = config.round if config.enable_crop else 1
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=config.num_workers)
    metrics = Metrics(model=model, device=device, num_classes=out_classes, val_loader=test_loader, rounds=rounds)

    log = ""
    if config.save_result:
        if not os.path.exists('../logs/predict'):
            os.makedirs('../logs/predict')
        log = "../logs/predict/find_best.txt"
        index = 1
        while os.path.exists(log):
            log = f"../logs/predict/find_best_{index}.txt"
            index += 1
        with open(log, 'a') as f:
            current_time = datetime.now()
            f.write(f"File saved time: {current_time}\n")
            f.write(f"Model dictionary: {config.model_dir}\n")
            f.write('-------------\n')

    model_dir = config.model_dir
    model_list = [
        f for f in os.listdir(model_dir)
        if f.endswith('.pth')
    ]
    best_model = ''
    best_acc = 0
    best_f1 = 0
    with tqdm(total=len(model_list), desc=f'Testing models', unit='model', position=0, leave=True) as pbar:
        for pth in sorted(model_list, key=lambda x: int(x.split('.')[0].split('epoch')[-1])):
            model_path = os.path.join(model_dir, pth)
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {model_path}')

            metrics.clear()
            pred_metric, cm = metrics.cal_metrics()
            logging.info(f'{rounds} rounds metrics:\n{pred_metric}')
            acc = pred_metric['accuracy']
            f1 = pred_metric['f1']
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1
                best_model = pth
            elif acc == best_acc and f1 > best_f1:
                best_acc = acc
                best_f1 = f1
                best_model = pth
            if config.save_result:
                with open(log, 'a') as f:
                    f.write(f'Model using: {model_path}\n')
                    f.write(f'Prediction metrics:\n')
                    for key, value in pred_metric.items():
                        f.write(f"    {key}: {value}\n")
                    f.write('-------------\n')
            pbar.update(1)
    with open(log, 'a') as f:
        f.write(f'Best model: {best_model}\n')
    logging.info(f"Results saved to: {log}\n")
    logging.info(f"Best model is {best_model}\n")
    model_path = os.path.join(model_dir, best_model)
    save_path = os.path.join(model_dir, 'best.pth')
    shutil.copy(model_path, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser.add_argument('--config', type=str, default='../config.yml', help='path of yaml file')
    parser.add_argument('--dir', type=str, default='../checkpoints/base', help='path of model file model')
    parser.add_argument('--dataset', type=str, default='../data/testing', help='path of test set folder')
    args = parser.parse_args()
    config = load_config(args.config)
    config.update_key('model_dir', args.dir)
    config.update_key('test_data_path', args.dataset)
    logging.info('Config loaded')

    find_best_model(config)
