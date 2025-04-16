import logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Utils.dataset import build_dataset
from Model import FLDNet
from Utils.evaluate import Metrics
import os
from datetime import datetime


def predict(config):
    device = config.device
    logging.info(f'Using device {device}')
    if config.load_model is None:
        raise ValueError('No model loaded for continue training')
    in_channels = config.input_channels
    out_classes = config.out_classes
    model = FLDNet(in_channels=in_channels, out_classes=out_classes, enable_se=config.enable_SE,
                   spp=config.enable_SPP, invert=config.invert).to(device)
    state_dict = torch.load(config.load_model，map_location=device)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {config.load_model}')
    transform = transforms.Compose([transforms.RandomCrop((160, 160))]) if config.enable_crop else None
    test_set, _ = build_dataset(img_dir=config.test_data_path, val_percent=0,
                                img_size=config.pred_size, transform=transform)

    rounds = config.round if config.enable_crop else 1
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=config.num_workers, pin_memory=True)
    metrics = Metrics(model=model, device=device, num_classes=out_classes, val_loader=test_loader, rounds=rounds)
    pred_metric, cm = metrics.cal_metrics()
    logging.info(f'{rounds} rounds metrics:\n{pred_metric}')

    if config.save_result:
        if not os.path.exists('./logs/predict'):
            os.makedirs('./logs/predict')
        log_list = os.listdir('./logs/predict')
        if len(log_list) == 0:
            max_num = 0
        else:
            log_list.sort(key=lambda x: int(x.split('.')[0]))
            max_num = int(log_list[-1].split('.')[0])
        log = f"./logs/predict/{max_num+1}.txt"
        with open(log, 'w') as f:
            current_time = datetime.now()
            f.write(f"File saved time: {current_time}\n")
            f.write(f'Model using: {config.load_model}\n')
            f.write(f'Prediction metrics:\n')
            for key, value in pred_metric.items():
                f.write(f"    {key}: {value}\n")
            f.write(f'Confusion matrix:\n')
            lines = [" ".join(map(str, row)) + "\n" for row in cm]
            f.writelines(lines)
        logging.info(f"Results saved to: {log}")
