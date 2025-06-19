
import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, preds, targets):
        """Update confusion matrix"""
        valid_mask = targets != self.ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for t, p in zip(targets.flatten(), preds.flatten()):
            self.confusion_matrix[t.long(), p.long()] += 1
    
    def compute_miou(self):
        """Compute mean IoU"""
        iou_per_class = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            
            if tp + fp + fn == 0:
                iou_per_class.append(0.0)
            else:
                iou_per_class.append(tp / (tp + fp + fn))
        
        return np.mean(iou_per_class)
    
    def compute_pixel_accuracy(self):
        """Compute pixel accuracy"""
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
7. Data Loading
Segmentation Dataset
python# datasets/segmentation_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image and mask paths
        self.images, self.masks = self._load_paths()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask