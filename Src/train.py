import logging
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import os
from Model import FLDNet
from Utils.dataset import build_dataset
from Utils.evaluate import Metrics
from datetime import datetime


def train(config):
    device = config.device
    logging.info(f'Using device {device}')

    # initialize model
    in_channels = config.input_channels
    out_classes = config.out_classes
    enable_se = config.enable_SE
    enable_spp = config.enable_SPP
    enable_invert = config.invert
    model = FLDNet(in_channels=in_channels, out_classes=out_classes, enable_se=enable_se,
                   spp=enable_spp, invert=enable_invert)
    model = model.to(device)
    if config.continue_train:
        if config.load_model is None:
            raise ValueError('No model loaded for continue training')
        state_dict = torch.load(config.load_model, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {config.load_model}')
    if device == 'cuda':
        torch.cuda.empty_cache()
    # Create train / validation set
    crop_size = config.crop_size
    transform = transforms.Compose([
        transforms.RandomCrop((crop_size, crop_size))
    ])
    dir_img = config.train_data_path
    val_percent = config.val_percent
    img_size = config.image_size
    batch_size = config.batch_size
    train_set, val_set = build_dataset(img_dir=dir_img, val_percent=val_percent, img_size=img_size,
                                       transform=transform)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True,
                            batch_size=1, num_workers=config.num_workers, pin_memory=True) if val_set is not None else None

    # Training initialize
    epochs = config.epoch
    learning_rate = config.lr
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.90, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_schedule_step, gamma=config.lr_schedule_gamma)
    elif config.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), learning_rate)
        scheduler = None
    else:
        raise ValueError(f'optimizer {config.optimizer} must be one of sgd or adamW')
    if out_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    cfg_info = f'''Model Configration:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(val_set) if val_set is not None else 0}
        Device:          {device}
        Patch size:      {img_size}
        Output classes:  {out_classes}
        Crop size:       {crop_size}
        SE Block:        {'enabled' if enable_se else 'disabled'}
        Inverted         {'enabled' if enable_invert else 'disabled'}
        SPP Block:       {'enabled' if enable_spp else 'disabled'}\n'''
    logging.info(cfg_info)

    global_step = 0
    log = None
    if config.save_log:
        if not os.path.exists('./logs/train'):
            os.makedirs('./logs/train')
        log_list = os.listdir('./logs/train')
        if len(log_list) == 0:
            max_num = 0
        else:
            log_list = [x for x in log_list if x.endswith('txt')]
            log_list.sort(key=lambda x: int(x.split('.')[0]))
            max_num = int(log_list[-1].split('.')[0])
        log = f'./logs/train/{max_num + 1}.txt'
        with open(log, 'a') as f:
            current_time = datetime.now()
            f.write(f'Strat training time: {current_time}\n')
            f.write(cfg_info)
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        train_acc = 0
        cnt = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', position=0, leave=False, unit='img') as pbar:
            for batch in train_loader:
                images, labels = batch
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                # Forward
                logits = model(images)
                # compute loss and accuracy in training
                correct = 0
                if out_classes != 1:
                    loss = criterion(logits, labels)
                    out = F.softmax(logits, dim=1)
                    _, pred = torch.max(out, 1)
                    correct += torch.eq(pred, labels).sum().item()
                else:
                    loss = criterion(logits.squeeze(1), labels.float())
                    out = F.sigmoid(logits.squeeze(1))
                    pred = out > 0.5
                    correct += (pred == labels).sum().item()
                batch_acc = correct / images.size(0)

                # optimize
                loss.backward()
                optimizer.step()
                global_step += 1
                scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                cnt += 1
                pbar.set_postfix(**{'loss': loss.item(), 'learning rate': current_lr, 'batch acc': batch_acc,
                                    'total step': global_step})
                
                
            epoch_loss = epoch_loss / cnt
            # Save model
            state_dict = model.state_dict()
            dir_checkpoint = config.model_dir
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
            torch.save(state_dict, os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch)))

            # Validation
            if val_loader is not None:
                metrics = Metrics(model=model, device=device, num_classes=out_classes, val_loader=val_loader)
                metric, _ = metrics.cal_metrics()
                acc = metric['accuracy']
                loss_val = metric['loss']
                epoch_result = f'Epoch:{epoch} | Validation acc:{acc} | Val Loss: {loss_val} | Train Loss: {epoch_loss}'
                logging.info(epoch_result)
                del metrics
            else:
                epoch_result = f'Epoch:{epoch} | Loss: {epoch_loss}'
                logging.info(epoch_result)

            if config.save_log:
                with open(log, 'a') as f:
                    f.write(epoch_result)
                    f.write(f"\n------------------\n")
