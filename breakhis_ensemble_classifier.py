import os
import re
import copy
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

BREAKHIS_ROOT = "./BreaKHis_v1/histology_slides/breast"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
NUM_WORKERS = 4
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "ensemble_checkpoints"

BACKBONES = ["resnet50", "resnext50_32x4d", "densenet201", "efficientnet_b3", "vgg19"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


FILENAME_PATTERN = re.compile(
    r"SOB_([BM])_([A-Z]+)-(\d+)-(\d+[A-Za-z]*)-(\d+)-(\d+)\.png", re.IGNORECASE
)


def parse_patient_id_and_label(filepath):
    fname = os.path.basename(filepath)
    m = FILENAME_PATTERN.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected BreakHis pattern: {fname}")
    b_or_m, subtype, year, patient_num, mag, seq = m.groups()
    label = 0 if b_or_m.upper() == "B" else 1
    patient_id = f"{b_or_m}_{subtype}_{year}_{patient_num}"
    return patient_id, label, subtype.upper()


def collect_all_images(root):
    exts = (".png", ".jpg", ".jpeg")
    return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]


def _stratify_key_with_fallback(patient_ids, strat_map):
    fine_keys = [strat_map[pid] for pid in patient_ids]
    counts = defaultdict(int)
    for k in fine_keys:
        counts[k] += 1
    if min(counts.values()) >= 2:
        return fine_keys
    print("  [warning] some (label, subtype) groups have <2 patients — "
          "falling back to label-only stratification for this split.")
    return [k.split("|")[0] for k in fine_keys]


def build_patient_wise_split(root, val_frac=0.15, test_frac=0.15):
    all_files = collect_all_images(root)
    if not all_files:
        raise FileNotFoundError(f"No images found under {root}. Update BREAKHIS_ROOT.")

    patient_to_files = defaultdict(list)
    patient_to_label = {}
    patient_to_subtype = {}
    for f in all_files:
        pid, label, subtype = parse_patient_id_and_label(f)
        patient_to_files[pid].append(f)
        patient_to_label[pid] = label
        patient_to_subtype[pid] = subtype

    patient_ids = list(patient_to_files.keys())
    strat_map = {pid: f"{patient_to_label[pid]}|{patient_to_subtype[pid]}" for pid in patient_ids}

    labels = [patient_to_label[pid] for pid in patient_ids]
    strat_keys = _stratify_key_with_fallback(patient_ids, strat_map)

    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_frac, stratify=strat_keys, random_state=SEED
    )
    train_val_strat_keys = _stratify_key_with_fallback(train_val_ids, strat_map)
    val_size_relative = val_frac / (1.0 - test_frac)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size_relative, stratify=train_val_strat_keys, random_state=SEED
    )

    def files_and_labels(pids):
        files, labs = [], []
        for pid in pids:
            for f in patient_to_files[pid]:
                files.append(f)
                labs.append(patient_to_label[pid])
        return files, labs

    train_files, train_labels = files_and_labels(train_ids)
    val_files, val_labels = files_and_labels(val_ids)
    test_files, test_labels = files_and_labels(test_ids)

    print(f"Patients -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Images   -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

    print("Per-subtype patient counts by split:")
    all_subtypes = sorted(set(patient_to_subtype.values()))
    for name, pids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        counts = {st: sum(1 for pid in pids if patient_to_subtype[pid] == st) for st in all_subtypes}
        print(f"  {name}: {counts}")

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


class BreakHisDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_dataloaders():
    (train_f, train_l), (val_f, val_l), (test_f, test_l) = build_patient_wise_split(BREAKHIS_ROOT)

    train_ds = BreakHisDataset(train_f, train_l, train_transform)
    val_ds = BreakHisDataset(val_f, val_l, eval_transform)
    test_ds = BreakHisDataset(test_f, test_l, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    n_benign = sum(1 for l in train_l if l == 0)
    n_malignant = sum(1 for l in train_l if l == 1)
    total = n_benign + n_malignant
    class_weights = torch.tensor(
        [total / (2.0 * n_benign), total / (2.0 * n_malignant)], dtype=torch.float32
    )
    return train_loader, val_loader, test_loader, class_weights


def build_model(arch_name):
    if arch_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    elif arch_name == "resnext50_32x4d":
        m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 2))
    elif arch_name == "densenet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, 2))
    elif arch_name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 2))
    elif arch_name == "vgg19":
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        in_features = m.classifier[6].in_features
        m.classifier[6] = nn.Linear(in_features, 2)
    else:
        raise ValueError(f"Unknown backbone: {arch_name}")
    return m.to(DEVICE)


HEAD_MODULE_GETTER = {
    "resnet50": lambda m: m.fc,
    "resnext50_32x4d": lambda m: m.fc,
    "densenet201": lambda m: m.classifier,
    "efficientnet_b3": lambda m: m.classifier,
    "vgg19": lambda m: m.classifier[6],
}


def get_param_groups(model, arch_name, backbone_lr, head_lr):
    head_module = HEAD_MODULE_GETTER[arch_name](model)
    head_params = list(head_module.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def _average_state_dicts(state_dicts):
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state:
        if avg_state[key].dtype.is_floating_point:
            stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
            avg_state[key] = stacked.mean(dim=0).to(avg_state[key].dtype)
    return avg_state


def train_one_model(arch_name, train_loader, val_loader, class_weights, top_k_avg=3):
    print(f"\n{'='*60}\nTraining {arch_name}\n{'='*60}")
    model = build_model(arch_name)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    param_groups = get_param_groups(model, arch_name, BACKBONE_LR, HEAD_LR)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    top_checkpoints = []
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{arch_name}_best.pt")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc, _, _, _ = evaluate(model, val_loader, criterion)
        train_acc = correct / total
        print(f"[{arch_name}] epoch {epoch+1}/{NUM_EPOCHS} train_acc {train_acc:.4f} val_acc {val_acc:.4f}")

        top_checkpoints.append((val_acc, copy.deepcopy(model.state_dict())))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        top_checkpoints = top_checkpoints[:top_k_avg]

    best_state_dicts = [sd for _, sd in top_checkpoints]
    averaged_state = _average_state_dicts(best_state_dicts)
    model.load_state_dict(averaged_state)
    torch.save(averaged_state, ckpt_path)

    final_val_acc, _, _, _ = evaluate(model, val_loader, criterion)
    top_k_accs = [round(a, 4) for a, _ in top_checkpoints]
    print(f"[{arch_name}] top-{top_k_avg} epoch val_accs averaged: {top_k_accs} "
          f"-> averaged-weights val_acc: {final_val_acc:.4f} (saved to {ckpt_path})")
    return model, final_val_acc


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_logits, all_labels = [], []
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        if criterion is not None:
            running_loss += criterion(outputs, labels).item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        all_logits.append(outputs.cpu())
        all_labels.append(labels.cpu())
    acc = correct / total
    loss = running_loss / total if criterion is not None else None
    return acc, loss, torch.cat(all_logits), torch.cat(all_labels)


def soft_vote(list_of_logits, weights=None):
    probs = [torch.softmax(logits, dim=1) for logits in list_of_logits]
    if weights is None:
        weights = [1.0] * len(probs)
    weights = np.array(weights) / np.sum(weights)
    combined = sum(w * p for w, p in zip(weights, probs))
    return combined


def expected_calibration_error(probs, correct, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - probs[in_bin].mean())
    return ece


def report_accuracy_and_ece(probs, labels, name):
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (preds == labels).astype(np.float32)
    acc = correct.mean()
    ece = expected_calibration_error(confidences, correct)
    print(f"{name:30s} accuracy: {acc:.4f}   ECE: {ece:.4f}")
    return acc, ece


class TemperatureScaler(nn.Module):
    MIN_T = 0.5
    MAX_T = 3.0
    REG_LAMBDA = 0.05

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, probs):
        logits = torch.log(probs.clamp(min=1e-8))
        return torch.softmax(logits / self.temperature, dim=1)

    def fit(self, val_probs, val_labels, max_iter=50, lr=0.01):
        val_probs = val_probs.to(DEVICE)
        val_labels = val_labels.to(DEVICE)
        nll = nn.NLLLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            log_probs = torch.log(self.forward(val_probs).clamp(min=1e-8))
            loss = nll(log_probs, val_labels) + self.REG_LAMBDA * (self.temperature - 1.0) ** 2
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(self.MIN_T, self.MAX_T)
        print(f"Fitted ensemble temperature: {self.temperature.item():.4f} "
              f"(clamped to [{self.MIN_T}, {self.MAX_T}], regularized toward 1.0)")
        return self


def fit_temperature_robust(val_probs, val_labels, n_resamples=10, resample_frac=0.8):
    n = val_probs.shape[0]
    fitted_temps = []
    for i in range(n_resamples):
        idx = torch.randperm(n)[: int(n * resample_frac)]
        scaler = TemperatureScaler().to(DEVICE)
        scaler.fit(val_probs[idx], val_labels[idx])
        fitted_temps.append(scaler.temperature.item())

    avg_temp = float(np.mean(fitted_temps))
    avg_temp = min(max(avg_temp, TemperatureScaler.MIN_T), TemperatureScaler.MAX_T)
    print(f"Temperature estimates across {n_resamples} resamples: "
          f"{[round(t, 3) for t in fitted_temps]}")
    print(f"Averaged, clamped temperature: {avg_temp:.4f}")

    final_scaler = TemperatureScaler().to(DEVICE)
    with torch.no_grad():
        final_scaler.temperature.fill_(avg_temp)
    return final_scaler


def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders()

    trained_models = {}
    val_accs = {}
    val_logits_per_model = {}
    test_logits_per_model = {}
    val_labels_ref = None
    test_labels_ref = None

    for arch in BACKBONES:
        model, val_acc = train_one_model(arch, train_loader, val_loader, class_weights)
        trained_models[arch] = model
        val_accs[arch] = val_acc

        v_acc, _, v_logits, v_labels = evaluate(model, val_loader)
        t_acc, _, t_logits, t_labels = evaluate(model, test_loader)
        val_logits_per_model[arch] = v_logits
        test_logits_per_model[arch] = t_logits
        val_labels_ref = v_labels
        test_labels_ref = t_labels

        print(f"[{arch}] standalone test accuracy: {t_acc:.4f}")

    print(f"\n{'='*60}\nEnsemble results\n{'='*60}")

    for arch in BACKBONES:
        probs = torch.softmax(test_logits_per_model[arch], dim=1).numpy()
        report_accuracy_and_ece(probs, test_labels_ref.numpy(), f"{arch} (individual)")

    ensemble_test_probs_unweighted = soft_vote(list(test_logits_per_model.values())).numpy()
    report_accuracy_and_ece(ensemble_test_probs_unweighted, test_labels_ref.numpy(), "Ensemble (unweighted)")

    weights = [val_accs[arch] for arch in BACKBONES]
    ensemble_test_probs_weighted = soft_vote(list(test_logits_per_model.values()), weights=weights).numpy()
    report_accuracy_and_ece(ensemble_test_probs_weighted, test_labels_ref.numpy(), "Ensemble (val-acc weighted)")

    ensemble_val_probs_weighted = soft_vote(list(val_logits_per_model.values()), weights=weights)
    scaler = fit_temperature_robust(ensemble_val_probs_weighted.to(DEVICE), val_labels_ref.to(DEVICE))

    calibrated_test_probs = scaler.forward(
        torch.tensor(ensemble_test_probs_weighted).to(DEVICE)
    ).detach().cpu().numpy()
    report_accuracy_and_ece(calibrated_test_probs, test_labels_ref.numpy(), "Ensemble (weighted + calibrated)")

    torch.save({
        "val_accs": val_accs,
        "temperature": scaler.temperature.detach().cpu(),
        "backbones": BACKBONES,
    }, os.path.join(CHECKPOINT_DIR, "ensemble_summary.pt"))
    print(f"\nAll checkpoints and ensemble summary saved in ./{CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()
