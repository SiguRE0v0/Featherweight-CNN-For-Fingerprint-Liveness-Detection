import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class Metrics:
    def __init__(self, model, num_classes, device, val_loader, rounds=1):
        self.classes = num_classes
        self.device = device
        self.model = model
        self.val_loader = val_loader
        self.result = []
        self.memory_saved = False
        self.threshold = 0.5
        self.rounds = rounds
        self.cm = []
        if num_classes == 1:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.loss = 0

    @torch.inference_mode()
    def run(self):
        self.model.eval()
        num_val = len(self.val_loader.dataset)
        loss = 0
        with torch.no_grad():
            with tqdm(total=num_val*self.rounds, desc=f'predicting {self.rounds} round(s)', unit='img', position=0, leave=False) as pbar:
                for i in range(self.rounds):
                    tmp_result = []
                    for batch in self.val_loader:
                        images, labels = batch
                        images = images.to(device=self.device, dtype=torch.float32)
                        labels = labels.to(device=self.device, dtype=torch.long)
                        logits = self.model(images)
                        loss += self.criterion(logits.squeeze(1), labels.float()) if self.classes == 1 else self.criterion(logits, labels)
                        tmp_result.append((logits.cpu().detach(), labels.cpu().detach()))
                        pbar.update(images.shape[0])
                    self.result.append(tmp_result)
        self.memory_saved = True
        self.loss = loss / (num_val*self.rounds)
        self.model.train()

    def cal_confusion_matrix(self):
        # cm: [[TN, FP],
        #      [FN, TP]]
        if not self.memory_saved:
            self.run()
        cm_size = self.classes if self.classes > 1 else 2
        cm = [[0] * cm_size for _ in range(cm_size)]
        for result in self.result:
            for logit, label in result:
                if self.classes != 1:
                    probs = F.softmax(logit, dim=1)
                    _, preds = torch.max(probs, 1)
                else:
                    probs = torch.sigmoid(logit)
                    preds = (probs > self.threshold).long()
                    preds = preds.squeeze(1)
                for true_label, pred_label in zip(label.tolist(), preds.tolist()):
                    cm[true_label][pred_label] += 1
        self.cm = cm
        return cm

    def cal_metrics(self):
        metrics = {}
        if not self.cm:
            self.cal_confusion_matrix()
        cm = np.array(self.cm)
        n_classes = self.classes
        loss = self.loss
        metrics['loss'] = loss.item()
        metrics['accuracy'] = np.trace(cm) / np.sum(cm)
        metrics['ACE'] = 1 - (np.trace(cm) / np.sum(cm))
        if n_classes <= 2:
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        else:
            macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
            total_tp, total_fp, total_fn = 0, 0, 0

            for i in range(n_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                macro_precision += precision
                macro_recall += recall
                macro_f1 += f1

                total_tp += tp
                total_fp += fp
                total_fn += fn

            macro_precision /= n_classes
            macro_recall /= n_classes
            macro_f1 /= n_classes

            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
            metrics.update({
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1
            })
        return metrics, cm

    def clear(self):
        self.cm = []
        self.result = []
        self.memory_saved = False
        self.loss = 0
