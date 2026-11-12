"""Turn CIFAR-10 into images/ + captions.csv for train.py.
Usage: python prepare_cifar.py --out cifar_data
"""
import argparse, os
import pandas as pd
from torchvision.datasets import CIFAR10

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cifar_data")
a = ap.parse_args()

ds = CIFAR10(root=os.path.join(a.out, "raw"), train=True, download=True)
img_dir = os.path.join(a.out, "images")
os.makedirs(img_dir, exist_ok=True)

rows = []
for i, (img, y) in enumerate(ds):
    name = f"{i:05d}.png"
    img.save(os.path.join(img_dir, name))
    rows.append({"image": name, "caption": f"a photo of a {ds.classes[y]}"})

pd.DataFrame(rows).to_csv(os.path.join(a.out, "captions.csv"), index=False)
print(f"saved {len(rows)} images + captions.csv to {a.out}")
