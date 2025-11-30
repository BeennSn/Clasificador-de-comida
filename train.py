import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prep_df(path: str, img_root: str = './food-101/images') -> pd.DataFrame:
    # Read lines and construct full paths using the class subdirectory
    raw_lines = open(path, 'r', encoding='utf-8').read().splitlines()
    lines = [l.strip() for l in raw_lines if l.strip()]
    labels = []
    files = []
    paths = []
    img_root_p = Path(img_root)
    for l in lines:
        parts = l.split('/')
        # expected format: <class>/<image_id> (without extension)
        if len(parts) >= 2:
            label = parts[0]
            fname = parts[1]
        else:
            # fallback: whole line as filename
            label = ''
            fname = parts[0]
        # ensure extension
        if not fname.lower().endswith('.jpg') and not fname.lower().endswith('.jpeg'):
            fname_with_ext = fname + '.jpg'
        else:
            fname_with_ext = fname
        # build path including the class subdirectory when available
        if label:
            p = img_root_p / label / fname_with_ext
        else:
            p = img_root_p / fname_with_ext
        labels.append(label)
        files.append(fname)
        paths.append(str(p))
    df = pd.DataFrame({'label': labels, 'file': files, 'path': paths})
    # filter out missing files early and warn the user
    exists_mask = df['path'].apply(lambda p: Path(p).exists())
    missing_count = (~exists_mask).sum()
    if missing_count > 0:
        print(f'Warning: {missing_count} image paths from {path} were not found under {img_root} and will be removed from the dataset.')
    df = df[exists_mask].sample(frac=1).reset_index(drop=True)
    return df


class Food21(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.path
        # explicit and clear error if image missing (helps DataLoader worker debug)
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        try:
            label_idx = self.classes.index(row.label)
        except ValueError:
            label_idx = len(self.classes) - 1
        return img, label_idx


def build_model(num_outputs=21, pretrained_backbone=True):
    # backbone
    weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    backbone = models.densenet201(weights=weights)

    # freeze backbone initially
    for p in backbone.parameters():
        p.requires_grad = False

    # classifier head matching previous script
    classifier = nn.Sequential(
        nn.Linear(1920, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 101),
    )
    backbone.classifier = classifier

    head = nn.Linear(101, num_outputs)

    model = nn.Sequential(backbone, head)
    return model


def train(args):
    # classes (first 20 + other)
    classes = open('./food-101/meta/classes.txt', 'r', encoding='utf-8').read().splitlines()
    classes_21 = classes[:20] + ['other']

    train_df = prep_df('./food-101/meta/train.txt')
    test_df = prep_df('./food-101/meta/test.txt')

    if args.train_subset < 1.0:
        n = int(len(train_df) * args.train_subset)
        train_df = train_df.sample(n)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = Food21(train_df, classes_21, transform=train_transforms)
    test_ds = Food21(test_df, classes_21, transform=test_transforms)

    # Optionally use a weighted sampler to mitigate class imbalance
    if args.use_sampler:
        label_counts = train_df['label'].value_counts().reindex(classes_21, fill_value=0)
        # avoid zero division
        label_counts = label_counts.replace(0, 1)
        class_weights = 1.0 / label_counts
        sample_weights = train_df['label'].map(lambda l: float(class_weights.get(l, class_weights.iloc[-1])))
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights.values, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    model = build_model(num_outputs=len(classes_21), pretrained_backbone=args.pretrained_backbone)
    model = model.to(DEVICE)

    # optionally load existing checkpoint to initialize backbone weights
    if args.checkpoint and Path(args.checkpoint).exists():
        print('Loading checkpoint', args.checkpoint)
        ck = torch.load(args.checkpoint, map_location='cpu')
        # if it's a state_dict
        if isinstance(ck, dict) and ('state_dict' in ck or 'model_state_dict' in ck or any(k.startswith('features.') for k in ck.keys())):
            sd = ck.get('state_dict', ck.get('model_state_dict', ck))
            # strip module. if present
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            # try load into backbone first
            try:
                model[0].load_state_dict(sd, strict=False)
                print('Loaded checkpoint into backbone (partial)')
            except Exception as e:
                print('Could not load checkpoint into backbone:', e)

    # unfreeze requested parts
    if args.unfreeze in ('classifier', 'head', 'all'):
        # classifier params
        for p in model[0].classifier.parameters():
            p.requires_grad = True
    if args.unfreeze in ('head', 'all'):
        for p in model[1].parameters():
            p.requires_grad = True
    if args.unfreeze == 'all':
        for p in model[0].parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # optional class weighting for loss (alternative to sampler)
    if args.use_class_weights:
        label_counts = train_df['label'].value_counts().reindex(classes_21, fill_value=0)
        label_counts = label_counts.replace(0, 1)
        class_weights = torch.tensor((1.0 / label_counts).values, dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

    # scheduler
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    best_acc = 0.0
    start_epoch = 0
    patience_counter = 0

    # optionally resume training from a saved checkpoint dict
    if args.resume and args.checkpoint and Path(args.checkpoint).exists():
        print(f'Resuming from checkpoint {args.checkpoint}')
        ck = torch.load(args.checkpoint, map_location='cpu')
        if 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer_state_dict'])
            except Exception as e:
                print('Warning: could not load optimizer state:', e)
        start_epoch = ck.get('epoch', 0) + 1
        best_acc = ck.get('best_acc', 0.0)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()

    # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Validation'):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        avg_loss = running_loss/len(train_loader) if len(train_loader) > 0 else 0.0
        print(f'Epoch {epoch+1} val acc: {acc:.4f} loss: {avg_loss:.4f}')
        if acc > best_acc:
            best_acc = acc
            out_path = args.output or 'food_classifier.pt'
            # save a checkpoint dict with optimizer and metadata
            ck = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': classes_21,
            }
            torch.save(ck, out_path)
            print('Saved best model to', out_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # step scheduler
        if scheduler is not None:
            scheduler.step()

        # early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f'Early stopping triggered after {patience_counter} epochs without improvement')
            break


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--unfreeze', choices=['none', 'classifier', 'head', 'all'], default='classifier')
    p.add_argument('--checkpoint', type=str, default='./food_classifier.pt')
    p.add_argument('--output', type=str, default='./food_classifier.pt')
    p.add_argument('--train-subset', type=float, default=0.3)
    p.add_argument('--pretrained-backbone', action='store_true')
    p.add_argument('--num-workers', type=int, default=4, help='DataLoader num_workers (set 0 on Windows for easier debugging)')
    p.add_argument('--pin-memory', action='store_true', help='Set pin_memory on DataLoader when using GPU')
    p.add_argument('--use-sampler', action='store_true', help='Use WeightedRandomSampler to balance classes')
    p.add_argument('--use-class-weights', action='store_true', help='Use class weights in loss instead of sampler')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--scheduler', choices=['none', 'step', 'cosine'], default='step')
    p.add_argument('--step-size', type=int, default=5)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=3, help='Early stopping patience (epochs)')
    p.add_argument('--resume', action='store_true', help='Resume training from --checkpoint if checkpoint is a full saved dict')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Training with args:', args)
    train(args)
