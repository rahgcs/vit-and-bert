import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

MEAN, STD = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)


def make_tf(img_size, train):
    if train:
        return T.Compose([T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.Resize(img_size), T.CenterCrop(img_size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class ImageCaptionDataset(Dataset):
    """One item per IMAGE (not per caption), so a batch never holds the same image
    twice (avoids false negatives in InfoNCE). Train: random caption; eval: first."""

    def __init__(self, df, root, img_size, train):
        self.groups = [(p, list(g["caption"])) for p, g in df.groupby("image")]
        self.root, self.train, self.tf = root, train, make_tf(img_size, train)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        path, caps = self.groups[i]
        img = Image.open(f"{self.root}/{path}").convert("RGB")
        return self.tf(img), random.choice(caps) if self.train else caps[0]


class Collate:
    """A class (not a nested function) so it can be pickled by DataLoader workers
    on Windows/macOS, which start workers with 'spawn'."""

    def __init__(self, tokenizer, max_len=32):
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, batch):
        imgs, caps = zip(*batch)
        tok = self.tokenizer(list(caps), padding=True, truncation=True,
                             max_length=self.max_len, return_tensors="pt")
        # identical caption text -> same label (positives for each other)
        uniq = {c: i for i, c in enumerate(dict.fromkeys(caps))}
        labels = torch.tensor([uniq[c] for c in caps])
        return torch.stack(imgs), tok["input_ids"], tok["attention_mask"], labels


def make_collate(tokenizer, max_len=32):
    return Collate(tokenizer, max_len)


def get_loaders(csv, root, tokenizer, img_size=224, bs=128, val_frac=0.05, workers=2, seed=0):
    """csv needs columns: image (relative path), caption (one row per caption)."""
    df = pd.read_csv(csv)
    imgs = df["image"].drop_duplicates().sample(frac=1.0, random_state=seed)
    n_val = max(1, int(len(imgs) * val_frac))
    val_set = set(imgs.iloc[:n_val])
    tr = ImageCaptionDataset(df[~df["image"].isin(val_set)], root, img_size, True)
    va = ImageCaptionDataset(df[df["image"].isin(val_set)], root, img_size, False)
    col = make_collate(tokenizer)
    return (DataLoader(tr, bs, shuffle=True, drop_last=True, num_workers=workers,
                       pin_memory=True, collate_fn=col),
            DataLoader(va, bs, shuffle=False, num_workers=workers, collate_fn=col))
