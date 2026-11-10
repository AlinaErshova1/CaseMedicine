import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .config import CFG
from .model import build_model
from .transforms import get_tta_transforms


@torch.no_grad()
def predict_with_tta(model, img_path, tta_transforms, class_names, device):
    """
    Усредняет softmax по всем TTA-трансформам.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    probs_sum = np.zeros(len(class_names), dtype=np.float64)

    for tf in tta_transforms:
        tensor = tf(image=img)["image"].unsqueeze(0).to(device)
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        probs_sum += probs

    probs_avg = probs_sum / len(tta_transforms)
    pred = int(probs_avg.argmax())
    return pred, probs_avg, img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    img_size = ckpt["img_size"]

    model, _, _ = build_model(ckpt["backbone"], len(class_names), CFG.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    if args.no_tta:
        from .transforms import get_val_transforms
        tta_transforms = [get_val_transforms(img_size)]
    else:
        tta_transforms = get_tta_transforms(img_size)

    pred, probs, img = predict_with_tta(model, args.image, tta_transforms,
                                        class_names, device)

    print(f"\nИзображение: {args.image}")
    print(f"TTA: {'off' if args.no_tta else f'{len(tta_transforms)}x'}")
    print(f"Предсказание: {class_names[pred]} "
          f"(уверенность {probs[pred] * 100:.2f}%)")
    print("Вероятности по классам:")
    for name, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:12s}: {p * 100:6.2f}%")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Pred: {class_names[pred]} ({probs[pred] * 100:.1f}%)")
    ax.axis("off")
    if args.save:
        plt.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"[DONE] Сохранено: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()