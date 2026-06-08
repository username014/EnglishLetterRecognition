import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

from utils.model_arch import ImprovedLetterCNN
from utils.augmentations import get_tta_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS_DIR = Path('checkpoints')
TEST_SIZE = 5000
NUM_CLASSES = 26


def load_test_data():
    ds = torchvision.datasets.EMNIST(
        root='./data', split='letters', train=False, download=True, transform=None
    )

    if len(ds) > TEST_SIZE:
        indices = torch.randperm(len(ds))[:TEST_SIZE]
        ds = Subset(ds, indices)

    return [(ds[i][0], ds[i][1] - 1) for i in range(len(ds))]


def predict_with_tta(model, pil_image, tta_transforms, device):
    model.eval()
    logits_list = []

    with torch.no_grad():
        for t in tta_transforms:
            img_tensor = t(pil_image).unsqueeze(0).to(device)
            logits_list.append(model(img_tensor))

    avg_logits = torch.stack(logits_list).mean(dim=0)
    return F.softmax(avg_logits, dim=1).squeeze()


def evaluate_model(model, test_data, device, tta_transforms):
    model.eval()
    correct, total = 0, 0
    confidences = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pil_img, label in test_data:
            probs = predict_with_tta(model, pil_img, tta_transforms, device)
            pred = probs.argmax().item()

            if pred == label:
                correct += 1
            total += 1

            confidences.append(probs.max().item())
            all_preds.append(pred)
            all_labels.append(label)

    accuracy = 100.0 * correct / total
    avg_confidence = np.mean(confidences) * 100

    ece = abs(np.mean(confidences) - (correct / total)) * 100

    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'avg_confidence': avg_confidence,
        'ece': ece
    }


def main():
    print(f"Evaluating models with TTA on {DEVICE}")

    tta_transforms = get_tta_transforms()
    print("Loading test data...")
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test samples.")

    results = []
    weight_files = sorted(CHECKPOINTS_DIR.glob("letter_cnn_*.pth"))

    if not weight_files:
        print("No weights found in checkpoints/")
        return

    for weight_path in weight_files:
        exp_name = weight_path.stem.replace("letter_cnn_", "")
        print(f"\nEvaluating: {exp_name}")

        model = ImprovedLetterCNN(num_classes=NUM_CLASSES).to(DEVICE)
        try:
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=False))
        except Exception as e:
            print(f"Error loading {weight_path}: {e}")
            continue

        metrics = evaluate_model(model, test_data, DEVICE, tta_transforms)

        score = 0.5 * metrics['accuracy'] + 0.3 * metrics['f1_macro'] + 0.2 * (100 - metrics['ece'])

        row = {
            'Model': exp_name,
            'Accuracy': f"{metrics['accuracy']:.2f}",
            'F1_Macro': f"{metrics['f1_macro']:.2f}",
            'Confidence': f"{metrics['avg_confidence']:.2f}",
            'ECE': f"{metrics['ece']:.2f}",
            'Composite_Score': f"{score:.2f}"
        }
        results.append(row)
        print(
            f"Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1_macro']:.2f}% | Conf: {metrics['avg_confidence']:.2f}% | Score: {score:.2f}")

    df = pd.DataFrame(results).sort_values('Composite_Score', ascending=False)

    print(df.to_string(index=False))

    output_file = "tta_evaluation.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved {output_file}")

    if not df.empty:
        best = df.iloc[0]
        print(f"\nBest model: {best['Model']} | Score: {best['Composite_Score']}")


if __name__ == "__main__":
    main()