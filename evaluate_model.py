import os
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import load_model_and_classes

ROOT = os.path.dirname(__file__)
META_TEST = os.path.join(ROOT, 'food-101', 'meta', 'test.txt')
IMAGES_ROOT = os.path.join(ROOT, 'food-101', 'images')


def prep_test_df(path: str, classes_21: List[str]) -> pd.DataFrame:
    """Prepare test dataframe filtering only the 21 trained classes."""
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    full_paths = [os.path.join(IMAGES_ROOT, l + '.jpg') for l in lines]
    labels = [l.split('/')[0] for l in lines]
    df = pd.DataFrame({'label': labels, 'path': full_paths})
    
    # Map labels: first 20 classes stay, rest -> 'other'
    first_20 = classes_21[:20]
    df['label'] = df['label'].apply(lambda x: x if x in first_20 else 'other')
    
    return df


test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Food21Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: List[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        # map label to index using classes list; unknown -> last index
        try:
            label_idx = self.classes.index(row.label)
        except ValueError:
            label_idx = len(self.classes) - 1
        return img, label_idx


def evaluate(checkpoint_path='./food_classifier.pt'):
    print('Loading model and classes...')
    model, classes, device = load_model_and_classes(checkpoint_path)

    print(f'Model loaded. Classes: {classes}')
    print('Preparing test dataset...')
    df = prep_test_df(META_TEST, classes)
    print(f'Test samples: {len(df)}, Label distribution:')
    print(df['label'].value_counts())
    
    dataset = Food21Dataset(df, classes, transform=test_transforms)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    top1_correct = 0
    top3_correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            top1 = torch.argmax(probs, dim=1)
            # top-3 check
            topk = torch.topk(probs, k=min(3, probs.size(1)), dim=1)

            top1_correct += (top1 == labels).sum().item()
            # check if labels in topk indices
            topk_idxs = topk.indices
            for i in range(labels.size(0)):
                if labels[i].item() in topk_idxs[i].tolist():
                    top3_correct += 1

            total += labels.size(0)
            all_preds.extend(top1.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    top1_acc = top1_correct / total
    top3_acc = top3_correct / total
    print(f'Total samples: {total}')
    print(f'Top-1 accuracy: {top1_acc:.4f}')
    print(f'Top-3 accuracy: {top3_acc:.4f}')

    # classification report and confusion matrix
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt

        print('\nClassification report (top-1):')
        print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))

        cm = confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        out_path = os.path.join(ROOT, 'confusion_matrix.png')
        fig.savefig(out_path, dpi=150)
        print(f'Confusion matrix saved to {out_path}')
    except Exception as e:
        print('sklearn/matplotlib not installed or failed to run report:', e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./food_classifier.pt')
    args = parser.parse_args()
    evaluate(args.checkpoint)
