import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import numpy as np
import csv
import sys
import itertools
import argparse
from pathlib import Path
from collections import Counter

from utils.model_arch import ImprovedLetterCNN
from utils.adversarial import AdversarialTTA, SelfAdversarialTrainer
from utils.augmentations import get_train_transforms, get_val_transforms, get_tta_transforms

BASE_CONFIG = {
    'data_root': './data',
    'batch_size': 128,
    'num_workers': 0 if sys.platform == 'win32' else 4,
    'num_classes': 26,
    'dropout_rate': 0.3,
    'epochs': 15,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'val_split': 0.1,
    'random_seed': 42,
}

WEIGHTS_DIR = Path('checkpoints')
MASTER_LOG = Path('training_comparison.csv')
NUM_CLASSES = 26

IMPROVEMENTS = [
    "sat_framework",
    "fgsm_fixed",
    "fgsm_adaptive",
    "pgd_fixed",
    "multi_adv",
    "tta"
]


def generate_experiments():
    experiments = []
    for r in range(1, len(IMPROVEMENTS) + 1):
        for combo in itertools.combinations(IMPROVEMENTS, r):
            config = BASE_CONFIG.copy()
            config["active_improvements"] = list(combo)
            config["name"] = "+".join(combo)

            config["use_sat"] = "sat_framework" in combo
            config["use_tta"] = "tta" in combo
            config["label_smoothing"] = 0.1 if "sat_framework" in combo else 0.0
            config["sat_prob"] = 0.6 if "sat_framework" in combo else 0.0

            attack_flags = [f for f in combo if f in ("fgsm_fixed", "fgsm_adaptive", "pgd_fixed")]
            if "pgd_fixed" in attack_flags:
                config["attack_type"] = "pgd"
                config["pgd_steps"] = 5
                config["epsilon"] = 0.03
                config["adaptive_epsilon"] = False
            elif "fgsm_adaptive" in attack_flags:
                config["attack_type"] = "fgsm"
                config["adaptive_epsilon"] = True
                config["eps_min"] = 0.01
                config["eps_max"] = 0.05
            elif "fgsm_fixed" in attack_flags:
                config["attack_type"] = "fgsm"
                config["adaptive_epsilon"] = False
                config["epsilon"] = 0.03
            else:
                if config["use_sat"]:
                    config["attack_type"] = "fgsm"
                    config["adaptive_epsilon"] = False
                    config["epsilon"] = 0.03

            config["n_adversarial"] = 3 if "multi_adv" in combo else 1
            experiments.append(config)
    return experiments


def get_weight_path(exp_cfg):
    safe_name = exp_cfg['name'].replace('+', '_').replace(' ', '_')
    return WEIGHTS_DIR / f"letter_cnn_{safe_name}.pth"


def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA unavailable. Running on CPU")
        return torch.device('cpu')
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    return torch.device('cuda')


def load_data(use_tta=False, batch_size=128):
    base_tf = transforms.ToTensor()
    full_ds = torchvision.datasets.EMNIST(
        root=BASE_CONFIG['data_root'], split='letters', train=True,
        download=True, transform=base_tf
    )
    val_size = int(len(full_ds) * BASE_CONFIG['val_split'])
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(BASE_CONFIG['random_seed'])
    )
    train_ds.dataset.transform = get_train_transforms()
    val_ds.dataset.transform = get_val_transforms()

    val_ds_raw = torchvision.datasets.EMNIST(
        root=BASE_CONFIG['data_root'], split='letters', train=True,
        download=False, transform=None
    )
    val_ds_raw = torch.utils.data.Subset(val_ds_raw, list(range(val_size, len(val_ds_raw))))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=BASE_CONFIG['num_workers'],
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=BASE_CONFIG['num_workers'],
                            pin_memory=pin)
    return train_loader, val_loader, val_ds_raw


def validate_standard(model, loader, device):
    model.eval()
    correct, total = 0, 0
    class_correct, class_total = torch.zeros(NUM_CLASSES), torch.zeros(NUM_CLASSES)
    confusion = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            labels = labels - 1
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels, preds):
                class_total[t] += 1
                if t == p:
                    class_correct[t] += 1
                else:
                    confusion[f"cls_{t}->cls_{p}"] += 1

    acc = 100.0 * correct / total
    class_acc = {i: (class_correct[i] / class_total[i]).item() * 100 if class_total[i] > 0 else 0 for i in
                 range(NUM_CLASSES)}
    return acc, class_acc, confusion.most_common(3)


def validate_with_adv_tta(model, loader, device, exp_cfg, batch_size=128):
    adv_tta = AdversarialTTA(
        model=model,
        device=device,
        epsilon=exp_cfg.get('epsilon', 0.03),
        n_adv=exp_cfg.get('n_adversarial', 3),
        attack_type=exp_cfg.get('attack_type', 'fgsm'),
        pgd_steps=exp_cfg.get('pgd_steps', 5)
    )

    model.eval()
    correct, total = 0, 0
    class_correct, class_total = torch.zeros(NUM_CLASSES), torch.zeros(NUM_CLASSES)
    confusion = Counter()

    for imgs, labels in loader:
        labels = labels - 1
        imgs, labels = imgs.to(device), labels.to(device)

        avg_logits = adv_tta.predict(imgs)
        preds = avg_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for t, p in zip(labels, preds):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
            else:
                confusion[f"cls_{t}->cls_{p}"] += 1

    acc = 100.0 * correct / total
    class_acc = {i: (class_correct[i] / class_total[i]).item() * 100 if class_total[i] > 0 else 0 for i in
                 range(NUM_CLASSES)}
    return acc, class_acc, confusion.most_common(3)


def run_experiment(exp_cfg, master_writer):
    use_sat = exp_cfg.get("use_sat", False)
    use_tta = exp_cfg.get("use_tta", False)
    method_tag = "SAT+TTA" if (use_sat and use_tta) else ("SAT" if use_sat else ("TTA" if use_tta else "Baseline"))

    print(f"Experiment [{len(exp_cfg['active_improvements'])} improvements]: {exp_cfg['name']}")
    print(f"Active {', '.join(exp_cfg['active_improvements'])}")
    device = check_gpu()
    torch.manual_seed(BASE_CONFIG['random_seed'])

    train_loader, val_loader, val_ds_raw = load_data(use_tta=use_tta, batch_size=exp_cfg['batch_size'])
    model = ImprovedLetterCNN(num_classes=NUM_CLASSES, dropout_rate=BASE_CONFIG['dropout_rate']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=exp_cfg['lr'], weight_decay=exp_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp_cfg['epochs'])
    sat_trainer = SelfAdversarialTrainer(model, optimizer, device, sat_config=exp_cfg)

    WEIGHTS_DIR.mkdir(exist_ok=True)
    weight_path = get_weight_path(exp_cfg)

    best_acc = 0.0
    tta_transforms = get_tta_transforms() if use_tta else None

    for epoch in range(exp_cfg['epochs']):
        model.train()
        total_loss, batches = 0, 0
        for imgs, labels in train_loader:
            loss = sat_trainer.train_step(imgs, labels - 1, epoch, exp_cfg['epochs'])
            total_loss += loss
            batches += 1

        avg_loss = total_loss / batches if batches > 0 else float('inf')

        if use_tta:
            val_acc, class_acc, top_confused = validate_with_adv_tta(model, val_loader, device, exp_cfg)
        else:
            val_acc, class_acc, top_confused = validate_standard(model, val_loader, device)

        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weight_path)

        log_row = {
            'Experiment': exp_cfg['name'],
            'Method': method_tag,
            'Epoch': epoch + 1,
            'Train_Loss': f"{avg_loss:.4f}",
            'Val_Acc_%': f"{val_acc:.2f}",
            'Best_Val_Acc_%': f"{best_acc:.2f}",
            'Top_Errors': "; ".join([f"{p} ({c})" for p, c in top_confused]) if top_confused else "None",
            'Lowest_Class_Acc': f"{min(class_acc.values()):.1f}%"
        }
        master_writer.writerow(log_row)
        print(
            f"  Epoch {epoch + 1:2d}/{exp_cfg['epochs']} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"Completed {exp_cfg['name']} | Weights {weight_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline with resume support")
    parser.add_argument('--resume', action='store_true', help='Skip experiments that already have saved weights')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start execution from specific experiment index (0-based)')
    args = parser.parse_args()

    print("Generating full factorial design (2^6 - 1 = 63 experiments)")
    print("Mutually exclusive attack strategies are resolved by priority: PGD > Adaptive > Fixed.")

    EXPERIMENTS = generate_experiments()
    print(f"Total experiments generated: {len(EXPERIMENTS)}\n")

    print("Experiment Index Map:")
    for i, exp in enumerate(EXPERIMENTS):
        print(f"  [{i:2d}] {exp['name']}")
    print()

    start_idx = 0
    if args.resume:
        for i, exp in enumerate(EXPERIMENTS):
            if not get_weight_path(exp).exists():
                start_idx = i
                break
        else:
            start_idx = len(EXPERIMENTS)
        print(f"First incomplete experiment with index {start_idx}")
    else:
        start_idx = args.start_from
        if start_idx < 0 or start_idx >= len(EXPERIMENTS):
            print(f"--start-from must be between 0 and {len(EXPERIMENTS) - 1}")
            sys.exit(1)
        print(f"Starting from index {start_idx}: {EXPERIMENTS[start_idx]['name']}")

    csv_mode = 'w' if start_idx == 0 else 'a'
    MASTER_LOG.parent.mkdir(exist_ok=True)

    with open(MASTER_LOG, csv_mode, newline='', encoding='utf-8') as f:
        fieldnames = [
            'Experiment', 'Method', 'Epoch', 'Train_Loss', 'Val_Acc_%',
            'Best_Val_Acc_%', 'Top_Errors', 'Lowest_Class_Acc'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if start_idx == 0:
            writer.writeheader()

        for i, exp in enumerate(EXPERIMENTS[start_idx:], start=start_idx):
            weight_path = get_weight_path(exp)
            if weight_path.exists():
                print(f"[{i:2d}] SKIP: {exp['name']} (weights already exist)")
                continue
            run_experiment(exp, writer)

    print(f"\nAll results saved/updated to {MASTER_LOG.resolve()}")