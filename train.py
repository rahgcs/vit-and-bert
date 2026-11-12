import argparse, math, os
import torch
from transformers import AutoTokenizer
from model import DualEncoder, clip_loss
from data import get_loaders


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    I, Tx, caps = [], [], []
    for img, ids, mask, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            I.append(model.encode_image(img.to(dev)).float())
            Tx.append(model.encode_text(ids.to(dev), mask.to(dev)).float())
        caps.append(ids)
    I, Tx = torch.cat(I), torch.cat(Tx)
    # global label = identical caption (token ids) -> a hit is any retrieved item
    # with the same caption as the query's ground truth (exact match for unique captions)
    L = max(c.size(1) for c in caps)
    flat = torch.cat([torch.nn.functional.pad(c, (0, L - c.size(1))) for c in caps])
    lab = torch.unique(flat, dim=0, return_inverse=True)[1].to(dev)
    sim = I @ Tx.t()
    out = {}
    for k in (1, 5, 10):
        top = sim.topk(k, 1).indices
        out[f"i2t@{k}"] = (lab[top] == lab[:, None]).any(1).float().mean().item()
        top = sim.t().topk(k, 1).indices
        out[f"t2i@{k}"] = (lab[top] == lab[:, None]).any(1).float().mean().item()
    model.train()
    return out


def set_text_trainable(model, flag):
    for p in model.text.bert.parameters():
        p.requires_grad = flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--text", default="distilbert-base-uncased")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=3, help="BERT frozen for first N epochs")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bert_lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--label_aware", action="store_true",
                    help="same-caption items are positives (use for class-label captions, e.g. CIFAR)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.text)
    tr, va = get_loaders(a.csv, a.root, tok, a.img_size, a.bs)
    model = DualEncoder(text_name=a.text, img_size=a.img_size, patch=a.patch).to(dev)

    bert_p = list(model.text.bert.parameters())
    ids = {id(p) for p in bert_p}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr},
                             {"params": bert_p, "lr": a.bert_lr}], weight_decay=a.wd)
    total = a.epochs * len(tr)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")   # T4: fp16, no bf16

    start = 0
    if os.path.exists(a.ckpt):                                  # resume (Colab drops sessions)
        c = torch.load(a.ckpt, map_location=dev)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); start = c["epoch"] + 1
        print(f"resumed from epoch {start}")

    for ep in range(start, a.epochs):
        set_text_trainable(model, ep >= a.freeze_epochs)
        for i, (img, ids_, mask, lab) in enumerate(tr):
            img, ids_, mask = img.to(dev), ids_.to(dev), mask.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                ie, te, s = model(img, ids_, mask)
            loss = clip_loss(ie.float(), te.float(), s, lab.to(dev) if a.label_aware else None)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(tr)} loss {loss.item():.4f} scale {s.item():.1f}")
        print(f"== epoch {ep}", {k: round(v, 3) for k, v in evaluate(model, va, dev).items()})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep}, a.ckpt)


if __name__ == "__main__":
    main()
