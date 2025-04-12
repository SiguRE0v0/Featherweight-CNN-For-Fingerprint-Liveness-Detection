from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from Utils import preprocess, traversal
import numpy as np
from torchvision import transforms


class FingerDataset(Dataset):
    def __init__(self, img_list, label_list, img_size=160, augmentations=True, transform=None):
        self.img_size = img_size
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

        logging.info('preloading and preprocessing images...')
        self.images = []
        self.labels = []
        for img_path, label in tqdm(zip(self.img_list, self.label_list), total=len(self.label_list), desc=f'preprocess', leave=False):
            img = Image.open(img_path).convert('L')
            patches = preprocess.patching(img, self.img_size)
            if augmentations:
                for patch in patches:
                    self.images.append(patch)
                    self.labels.append(label)
                    patch_aug, label_aug = preprocess.augmentation(patch, label)
                    self.images.extend(patch_aug)
                    self.labels.extend(label_aug)
            else:
                self.images.append(patches[0])
                self.labels.append(label)
            img.close()
        logging.info(f'Finished creating dataset with {len(self.images)} images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image.float().contiguous(), label


def build_dataset(img_dir, val_percent, img_size=160, transform=None):
    img_list, label_list = traversal.file_traversal(img_dir)
    if val_percent == 0:
        dataset = FingerDataset(img_list=img_list, label_list=label_list, img_size=img_size,
                                augmentations=False, transform=transform)
        return dataset, None
    indices = np.random.permutation(len(img_list))
    train_indices = indices[int(val_percent * len(indices)):]
    val_indices = indices[:int(val_percent * len(indices))]

    train_set = FingerDataset(
        img_list=[img_list[i] for i in train_indices],
        label_list=[label_list[i] for i in train_indices],
        img_size=img_size,
        augmentations=True,
        transform=transform
    )
    val_set = FingerDataset(
        img_list=[img_list[i] for i in val_indices],
        label_list=[label_list[i] for i in val_indices],
        img_size=img_size,
        augmentations=False,
        transform=transform
    )
    return train_set, val_set
