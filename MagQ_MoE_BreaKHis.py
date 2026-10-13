# ================================================================
# MagQ-MoE: Magnification-Aware Quantum Mixture-of-Experts
# Dataset: BreaKHis | Platform: Kaggle | GPU: recommended
#
# Architecture:
# Image -> ImageNet-pretrained ResNet18 -> f in R^512
#                         |-> Router(f + magnification one-hot) -> g4,g6,g8
#                         |-> 4Q AQB-VQC expert -> logits_4
#                         |-> 6Q AQB-VQC expert -> logits_6
#                         |-> 8Q AQB-VQC expert -> logits_8
# Final logits = g4*logits_4 + g6*logits_6 + g8*logits_8
#
# Important:
# - Patient-wise split (same SEED=42 strategy as AQB work)
# - No silent fallback to random ResNet18 weights
# - Validation-only threshold calibration (Youden J)
# - Router/expert analysis by 40X/100X/200X/400X
# ================================================================

import os, re, gc, math, time, copy, random, shutil, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ================================================================
# 1. CONFIGURATION
# ================================================================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
TRAIN_RATIO = 0.70

EXPERT_QUBITS = (4, 6, 8)
VQC_DEPTH = 2
ROUTER_HIDDEN = 128
ROUTER_TEMPERATURE = 1.0

EPOCHS = 25
PATIENCE = 6
LR_BACKBONE = 1e-5
LR_HEAD = 2e-4
WEIGHT_DECAY = 1e-4
BALANCE_LAMBDA = 0.02       # prevents one expert from taking all samples
GRAD_CLIP = 5.0

OUTPUT_DIR = Path('/kaggle/working/MagQ_MoE_BreaKHis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAG_TO_IDX = {'40X': 0, '100X': 1, '200X': 2, '400X': 3}
IDX_TO_MAG = {v: k for k, v in MAG_TO_IDX.items()}

print('=' * 90)
print('MagQ-MoE: MAGNIFICATION-AWARE QUANTUM MIXTURE-OF-EXPERTS')
print('=' * 90)
print('Device:', DEVICE)
if torch.cuda.is_available():
    print('GPU   :', torch.cuda.get_device_name(0))
print('Output:', OUTPUT_DIR)

# ================================================================
# 2. REPRODUCIBILITY
# ================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# ================================================================
# 3. FIND BreaKHis IMAGES + METADATA
# ================================================================
INPUT_ROOT = Path('/kaggle/input')
all_images = []
for ext in ('*.png', '*.jpg', '*.jpeg'):
    all_images.extend(INPUT_ROOT.rglob(ext))

image_paths = [
    p for p in all_images
    if p.name.upper().startswith('SOB_')
    and (('benign' in str(p).lower()) or ('malignant' in str(p).lower()))
]

if not image_paths:
    raise RuntimeError('No BreaKHis SOB_ images found under /kaggle/input.')

def get_label(path):
    s = str(path).lower()
    if 'malignant' in s: return 1
    if 'benign' in s: return 0
    raise ValueError(f'Cannot determine label: {path}')

def get_patient_id(filename):
    stem = Path(filename).stem
    return re.sub(r'-(40|100|200|400)-\d+$', '', stem, flags=re.IGNORECASE)

def get_magnification(filename):
    m = re.search(r'-(40|100|200|400)-\d+\.', filename, flags=re.IGNORECASE)
    return (m.group(1) + 'X') if m else 'Unknown'

records = []
for p in image_paths:
    label = get_label(p)
    mag = get_magnification(p.name)
    records.append({
        'path': str(p),
        'filename': p.name,
        'label': label,
        'class': 'Malignant' if label else 'Benign',
        'patient': get_patient_id(p.name),
        'magnification': mag
    })

df = pd.DataFrame(records).drop_duplicates(subset=['path']).reset_index(drop=True)

if (df['magnification'] == 'Unknown').any():
    bad = df[df['magnification'] == 'Unknown'][['filename']].head()
    raise RuntimeError(f'Unknown magnification found:\n{bad}')

print('\nDataset summary')
print('Images  :', len(df))
print('Patients:', df['patient'].nunique())
print(df['class'].value_counts())
print('\nMagnification x class:')
print(pd.crosstab(df['magnification'], df['class']))
df.to_csv(OUTPUT_DIR / '00_dataset_inventory.csv', index=False)

# ================================================================
# 4. SAME PATIENT-WISE SPLIT STRATEGY AS P1
# ================================================================
patient_df = df.groupby('patient').agg(label=('label', 'first')).reset_index()
benign_patients = patient_df.loc[patient_df.label == 0, 'patient'].tolist()
malignant_patients = patient_df.loc[patient_df.label == 1, 'patient'].tolist()

def split_patient_class(patient_list, seed):
    tr, tmp = train_test_split(
        patient_list, test_size=(1.0 - TRAIN_RATIO), random_state=seed
    )
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed)
    return tr, va, te

b_tr, b_va, b_te = split_patient_class(benign_patients, SEED)
m_tr, m_va, m_te = split_patient_class(malignant_patients, SEED)

train_patients = set(b_tr + m_tr)
val_patients = set(b_va + m_va)
test_patients = set(b_te + m_te)
assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

train_df = df[df.patient.isin(train_patients)].reset_index(drop=True)
val_df = df[df.patient.isin(val_patients)].reset_index(drop=True)
test_df = df[df.patient.isin(test_patients)].reset_index(drop=True)

for name, part in [('TRAIN', train_df), ('VAL', val_df), ('TEST', test_df)]:
    print(f'{name:5s}: images={len(part):4d} patients={part.patient.nunique():2d} '
          f'benign={(part.label==0).sum():4d} malignant={(part.label==1).sum():4d}')

pd.concat([
    train_df.assign(split='train'),
    val_df.assign(split='validation'),
    test_df.assign(split='test')
]).to_csv(OUTPUT_DIR / '01_patient_wise_split.csv', index=False)

# ================================================================
# 5. TRANSFORMS + DATASET
# ================================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BreakHisMagDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(int(row.label), dtype=torch.long)
        mag_idx = torch.tensor(MAG_TO_IDX[row.magnification], dtype=torch.long)
        return image, label, mag_idx, idx

train_loader = DataLoader(
    BreakHisMagDataset(train_df, train_transform), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)
val_loader = DataLoader(
    BreakHisMagDataset(val_df, eval_transform), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)
test_loader = DataLoader(
    BreakHisMagDataset(test_df, eval_transform), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0)
)

# ================================================================
# 6. ROBUST PRETRAINED RESNET18 LOADER
#    First searches Kaggle inputs; otherwise tries Internet.
#    NEVER silently uses random weights.
# ================================================================
def load_pretrained_resnet18():
    candidates = glob.glob('/kaggle/input/**/resnet18-f37072fd.pth', recursive=True)
    if candidates:
        path = candidates[0]
        print('Loading local ImageNet ResNet18:', path)
        model = models.resnet18(weights=None)
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print('SUCCESS: pretrained ResNet18 loaded locally.')
        return model
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print('SUCCESS: pretrained ResNet18 loaded from torchvision.')
        return model
    except Exception as e:
        raise RuntimeError(
            'ImageNet-pretrained ResNet18 is mandatory. Enable Kaggle Internet or attach '
            'resnet18-f37072fd.pth as a Kaggle input. Original error: ' + str(e)
        )

# Verify now, before long training.
_tmp = load_pretrained_resnet18()
del _tmp
gc.collect()

# ================================================================
# 7. DIFFERENTIABLE QUANTUM STATEVECTOR SIMULATOR
# ================================================================
COMPLEX = torch.complex64

def rx_gate(theta):
    c, s = torch.cos(theta/2), torch.sin(theta/2)
    z = torch.zeros_like(c)
    r1 = torch.stack([torch.complex(c,z), torch.complex(z,-s)], dim=-1)
    r2 = torch.stack([torch.complex(z,-s), torch.complex(c,z)], dim=-1)
    return torch.stack([r1,r2], dim=-2)

def ry_gate(theta):
    c, s = torch.cos(theta/2), torch.sin(theta/2)
    z = torch.zeros_like(c)
    r1 = torch.stack([torch.complex(c,z), torch.complex(-s,z)], dim=-1)
    r2 = torch.stack([torch.complex(s,z), torch.complex(c,z)], dim=-1)
    return torch.stack([r1,r2], dim=-2)

def rz_gate(theta):
    h = theta/2
    em = torch.complex(torch.cos(h), -torch.sin(h))
    ep = torch.complex(torch.cos(h), torch.sin(h))
    z = torch.zeros_like(em)
    return torch.stack([torch.stack([em,z],-1), torch.stack([z,ep],-1)], -2)

class QuantumCircuitLayer(nn.Module):
    def __init__(self, n_qubits, depth=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.theta = nn.Parameter(0.05 * torch.randn(depth, n_qubits, 3))
        for i in range(n_qubits):
            self.register_buffer(
                f'cnot_perm_{i}', self._cnot_permutation(n_qubits, i, (i+1)%n_qubits)
            )

    @staticmethod
    def _cnot_permutation(n_qubits, control, target):
        size = 2**n_qubits
        perm = torch.arange(size, dtype=torch.long)
        cm = 1 << (n_qubits - 1 - control)
        tm = 1 << (n_qubits - 1 - target)
        for i in range(size):
            if i & cm:
                perm[i] = i ^ tm
        return perm

    def apply_single_gate(self, state, gate, qubit):
        B, q = state.shape[0], self.n_qubits
        tensor = state.reshape([B] + [2]*q)
        target_axis = qubit + 1
        axes = list(range(q+1))
        axes.pop(target_axis); axes.append(target_axis)
        tensor = tensor.permute(axes)
        original_shape = tensor.shape
        tensor = tensor.reshape(B, -1, 2)
        if gate.dim() == 3:
            tensor = torch.einsum('bni,bji->bnj', tensor, gate)
        else:
            tensor = torch.einsum('bni,ji->bnj', tensor, gate)
        tensor = tensor.reshape(original_shape)
        tensor = tensor.permute(np.argsort(axes).tolist())
        return tensor.reshape(B, 2**q)

    def forward(self, z):
        B, q = z.shape[0], self.n_qubits
        state = torch.zeros(B, 2**q, dtype=COMPLEX, device=z.device)
        state[:,0] = 1.0 + 0.0j

        for i in range(q):
            state = self.apply_single_gate(state, ry_gate(z[:,i]), i)

        for d in range(self.depth):
            for i in range(q):
                state = self.apply_single_gate(state, rx_gate(self.theta[d,i,0]), i)
                state = self.apply_single_gate(state, ry_gate(self.theta[d,i,1]), i)
                state = self.apply_single_gate(state, rz_gate(self.theta[d,i,2]), i)
            for i in range(q):
                state = state[:, getattr(self, f'cnot_perm_{i}')]

        probs = state.real.pow(2) + state.imag.pow(2)
        basis = torch.arange(2**q, device=z.device)
        exps = []
        for i in range(q):
            mask = 1 << (q - 1 - i)
            sign = torch.where((basis & mask) == 0, 1.0, -1.0)
            exps.append(torch.sum(probs * sign, dim=1))
        return torch.stack(exps, dim=1)

# ================================================================
# 8. QUANTUM EXPERT + MAGNIFICATION-AWARE ROUTER
# ================================================================
class QuantumExpert(nn.Module):
    def __init__(self, q, depth=2):
        super().__init__()
        self.q = q
        self.aqb = nn.Linear(512, q)
        self.norm = nn.LayerNorm(q)
        self.vqc = QuantumCircuitLayer(q, depth)
        self.classifier = nn.Linear(q, 2)

    def forward(self, features):
        z = self.aqb(features)
        z = self.norm(z)
        z_angle = math.pi * torch.tanh(z)
        h = self.vqc(z_angle)
        logits = self.classifier(h)
        return logits, h, z_angle

class MagQMoE(nn.Module):
    def __init__(self, expert_qubits=(4,6,8), depth=2, router_hidden=128, temperature=1.0):
        super().__init__()
        self.expert_qubits = tuple(expert_qubits)
        self.temperature = temperature

        backbone = load_pretrained_resnet18()
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Same transfer-learning policy as corrected P1: freeze all, fine-tune layer4.
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.experts = nn.ModuleList([
            QuantumExpert(q, depth) for q in self.expert_qubits
        ])

        # Explicit magnification awareness: 512 CNN features + 4-D one-hot magnification.
        self.router = nn.Sequential(
            nn.Linear(512 + 4, router_hidden),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(router_hidden, len(self.expert_qubits))
        )

    def forward(self, x, mag_idx):
        f = self.backbone(x)
        mag_onehot = F.one_hot(mag_idx, num_classes=4).float()
        router_input = torch.cat([f, mag_onehot], dim=1)
        router_logits = self.router(router_input)
        gates = torch.softmax(router_logits / self.temperature, dim=1)

        expert_logits = []
        expert_h = []
        for expert in self.experts:
            logits, h, _ = expert(f)
            expert_logits.append(logits)
            expert_h.append(h)

        # [B,E,2]
        stacked_logits = torch.stack(expert_logits, dim=1)
        mixed_logits = torch.sum(gates.unsqueeze(-1) * stacked_logits, dim=1)
        return mixed_logits, gates, stacked_logits, f, expert_h

# ================================================================
# 9. LOSSES + METRICS
# ================================================================
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = len(train_df) / (2.0 * class_counts)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
print('Class weights:', class_weights)

def load_balance_loss(gates):
    # KL(mean expert usage || uniform). Minimum = 0.
    mean_gate = gates.mean(dim=0).clamp_min(1e-8)
    uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
    return torch.sum(mean_gate * torch.log(mean_gate / uniform))

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = np.nan
    return {
        'Threshold': float(threshold),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity': sens,
        'Specificity': spec,
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': auc,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }

def validation_threshold(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob)
    valid = np.isfinite(th)
    fpr, tpr, th = fpr[valid], tpr[valid], th[valid]
    j = tpr - fpr
    return float(th[np.argmax(j)])

# ================================================================
# 10. TRAINING / PREDICTION
# ================================================================
@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, probs, idxs, mags, gates_all = [], [], [], [], []
    expert_probs = {q: [] for q in EXPERT_QUBITS}
    total_loss = 0.0

    for images, labels, mag_idx, indices in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        mag_idx_dev = mag_idx.to(DEVICE, non_blocking=True)

        logits, gates, stacked_logits, _, _ = model(images, mag_idx_dev)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        p = torch.softmax(logits, dim=1)[:,1]

        ys.extend(labels.cpu().numpy())
        probs.extend(p.cpu().numpy())
        idxs.extend(indices.numpy())
        mags.extend(mag_idx.numpy())
        gates_all.append(gates.cpu().numpy())

        for e, q in enumerate(EXPERT_QUBITS):
            ep = torch.softmax(stacked_logits[:,e,:], dim=1)[:,1]
            expert_probs[q].extend(ep.cpu().numpy())

    return {
        'loss': total_loss/len(loader.dataset),
        'y': np.asarray(ys),
        'prob': np.asarray(probs),
        'idx': np.asarray(idxs),
        'mag_idx': np.asarray(mags),
        'gates': np.concatenate(gates_all, axis=0),
        'expert_probs': {q: np.asarray(v) for q,v in expert_probs.items()}
    }

def train_model():
    seed_everything(SEED)
    model = MagQMoE(
        EXPERT_QUBITS, VQC_DEPTH, ROUTER_HIDDEN, ROUTER_TEMPERATURE
    ).to(DEVICE)

    backbone_params = list(model.backbone.layer4.parameters())
    head_params = (
        list(model.experts.parameters()) + list(model.router.parameters())
    )
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    print('\nTrainable parameters:', f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    history, best_state, best_auc, wait = [], None, -np.inf, 0

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        total_loss = total_ce = total_bal = 0.0
        y_train, p_train = [], []
        gate_sum = np.zeros(len(EXPERT_QUBITS), dtype=np.float64)
        n_seen = 0

        for images, labels, mag_idx, _ in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            mag_idx = mag_idx.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, gates, _, _, _ = model(images, mag_idx)
            ce = criterion(logits, labels)
            bal = load_balance_loss(gates)
            loss = ce + BALANCE_LAMBDA * bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bs = images.size(0)
            total_loss += loss.item()*bs
            total_ce += ce.item()*bs
            total_bal += bal.item()*bs
            prob = torch.softmax(logits.detach(), dim=1)[:,1]
            y_train.extend(labels.detach().cpu().numpy())
            p_train.extend(prob.cpu().numpy())
            gate_sum += gates.detach().sum(dim=0).cpu().numpy()
            n_seen += bs

        trm = compute_metrics(y_train, p_train, 0.5)
        val = predict(model, val_loader)
        vam = compute_metrics(val['y'], val['prob'], 0.5)
        scheduler.step(vam['AUC'])
        mean_gates = gate_sum / max(n_seen, 1)

        row = {
            'Epoch': epoch,
            'Train_Loss': total_loss/len(train_loader.dataset),
            'Train_CE': total_ce/len(train_loader.dataset),
            'Train_Balance': total_bal/len(train_loader.dataset),
            'Train_Accuracy': trm['Accuracy'], 'Train_AUC': trm['AUC'],
            'Val_Loss': val['loss'], 'Val_Accuracy': vam['Accuracy'],
            'Val_Balanced_Accuracy': vam['Balanced_Accuracy'], 'Val_AUC': vam['AUC'],
            **{f'Train_Gate_{q}Q': mean_gates[i] for i,q in enumerate(EXPERT_QUBITS)}
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | loss={row['Train_Loss']:.4f} | "
            f"trainAUC={trm['AUC']:.4f} | valAcc={vam['Accuracy']:.4f} | "
            f"valBAcc={vam['Balanced_Accuracy']:.4f} | valAUC={vam['AUC']:.4f} | "
            f"gates={np.round(mean_gates,3)} | {time.time()-t0:.1f}s"
        )

        if vam['AUC'] > best_auc:
            best_auc = vam['AUC']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print('Early stopping.')
                break

    model.load_state_dict(best_state)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(OUTPUT_DIR/'02_training_history.csv', index=False)
    torch.save({
        'model_state_dict': model.state_dict(), 'seed': SEED,
        'expert_qubits': EXPERT_QUBITS, 'best_val_auc': best_auc,
        'mag_to_idx': MAG_TO_IDX
    }, OUTPUT_DIR/'MagQ_MoE_best.pt')
    return model, hist_df

model, history_df = train_model()

# ================================================================
# 11. FINAL VALIDATION THRESHOLD + TEST RESULTS
# ================================================================
val_out = predict(model, val_loader)
opt_threshold = validation_threshold(val_out['y'], val_out['prob'])
test_out = predict(model, test_loader)

m05 = compute_metrics(test_out['y'], test_out['prob'], 0.5)
mopt = compute_metrics(test_out['y'], test_out['prob'], opt_threshold)

results_df = pd.DataFrame([
    {'Model':'MagQ-MoE', 'Evaluation':'Fixed 0.5', **m05},
    {'Model':'MagQ-MoE', 'Evaluation':'Validation Youden J', **mopt}
])
results_df.to_csv(OUTPUT_DIR/'03_final_test_metrics.csv', index=False)
print('\nFINAL TEST RESULTS')
print(results_df.round(4).to_string(index=False))
print(f'\nValidation-selected threshold = {opt_threshold:.6f}')

# ================================================================
# 12. SAVE SAMPLE-LEVEL PREDICTIONS + ROUTING
# ================================================================
pred_df = test_df.iloc[test_out['idx']].copy().reset_index(drop=True)
pred_df['probability_malignant'] = test_out['prob']
pred_df['prediction_05'] = (test_out['prob'] >= 0.5).astype(int)
pred_df['validation_threshold'] = opt_threshold
pred_df['prediction_calibrated'] = (test_out['prob'] >= opt_threshold).astype(int)

for i,q in enumerate(EXPERT_QUBITS):
    pred_df[f'gate_{q}Q'] = test_out['gates'][:,i]
    pred_df[f'expert_{q}Q_prob_malignant'] = test_out['expert_probs'][q]

hard_idx = np.argmax(test_out['gates'], axis=1)
pred_df['selected_expert'] = [f'{EXPERT_QUBITS[i]}Q' for i in hard_idx]

# Optional hard top-1 inference: use only the expert preferred by the router.
# This is useful for studying whether soft routing can later be converted to
# a resource-selective inference policy.
hard_prob = np.zeros(len(pred_df), dtype=np.float32)
for i, q in enumerate(EXPERT_QUBITS):
    mask = (hard_idx == i)
    hard_prob[mask] = test_out['expert_probs'][q][mask]
pred_df['hard_routed_probability_malignant'] = hard_prob
pred_df['hard_routed_prediction_05'] = (hard_prob >= 0.5).astype(int)
pred_df.to_csv(OUTPUT_DIR/'04_test_predictions_and_routing.csv', index=False)

hard_metrics = compute_metrics(test_out['y'], hard_prob, 0.5)
pd.DataFrame([{'Model':'Hard top-1 routed MagQ-MoE', **hard_metrics}]).to_csv(
    OUTPUT_DIR/'04b_hard_top1_routing_metrics.csv', index=False
)
print('\nHARD TOP-1 ROUTING (diagnostic/resource-selective inference)')
print(pd.DataFrame([hard_metrics]).round(4).to_string(index=False))

# ================================================================
# 13. MAGNIFICATION-SPECIFIC CLASSIFICATION PERFORMANCE
# ================================================================
mag_rows = []
for mag in ['40X','100X','200X','400X']:
    sub = pred_df[pred_df.magnification == mag]
    if len(sub) and sub.label.nunique() == 2:
        mm = compute_metrics(sub.label.values, sub.probability_malignant.values, opt_threshold)
        mag_rows.append({'Magnification':mag, 'N':len(sub), **mm})
mag_metrics_df = pd.DataFrame(mag_rows)
mag_metrics_df.to_csv(OUTPUT_DIR/'05_magnification_metrics.csv', index=False)
print('\nMAGNIFICATION-SPECIFIC PERFORMANCE')
print(mag_metrics_df.round(4).to_string(index=False))

# ================================================================
# 14. ROUTER ANALYSIS BY MAGNIFICATION
# ================================================================
route_rows = []
for mag in ['40X','100X','200X','400X']:
    sub = pred_df[pred_df.magnification == mag]
    row = {'Magnification': mag, 'N': len(sub)}
    for q in EXPERT_QUBITS:
        row[f'Mean_Gate_{q}Q'] = sub[f'gate_{q}Q'].mean()
        row[f'Hard_Select_{q}Q_Pct'] = 100.0*(sub.selected_expert == f'{q}Q').mean()
    route_rows.append(row)
routing_df = pd.DataFrame(route_rows)
routing_df.to_csv(OUTPUT_DIR/'06_routing_by_magnification.csv', index=False)
print('\nROUTING BY MAGNIFICATION')
print(routing_df.round(4).to_string(index=False))

# Also routing by true class and magnification.
route_detail = pred_df.groupby(['magnification','class'])[
    [f'gate_{q}Q' for q in EXPERT_QUBITS]
].mean().reset_index()
route_detail.to_csv(OUTPUT_DIR/'07_routing_by_magnification_and_class.csv', index=False)

# ================================================================
# 15. INDIVIDUAL EXPERT PERFORMANCE INSIDE THE TRAINED MoE
#     (diagnostic only; experts were jointly trained in the mixture)
# ================================================================
expert_rows = []
for q in EXPERT_QUBITS:
    em = compute_metrics(test_out['y'], test_out['expert_probs'][q], 0.5)
    expert_rows.append({'Component':f'{q}Q expert inside MoE', **em})
expert_rows.append({'Component':'MagQ-MoE mixture', **m05})
expert_df = pd.DataFrame(expert_rows)
expert_df.to_csv(OUTPUT_DIR/'08_internal_expert_vs_mixture.csv', index=False)
print('\nINTERNAL EXPERT VS MIXTURE (diagnostic)')
print(expert_df.round(4).to_string(index=False))

# ================================================================
# 16. PAPER-READY FIGURES
# ================================================================
# ROC
fpr, tpr, _ = roc_curve(test_out['y'], test_out['prob'])
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"MagQ-MoE AUC={m05['AUC']:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('MagQ-MoE ROC — BreaKHis'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUTPUT_DIR/'09_ROC_MagQ_MoE.png', dpi=300); plt.show(); plt.close()

# Confusion matrix at validation-calibrated threshold
cm = confusion_matrix(test_out['y'], (test_out['prob'] >= opt_threshold).astype(int))
plt.figure(figsize=(5,5)); plt.imshow(cm)
plt.xticks([0,1], ['Benign','Malignant']); plt.yticks([0,1], ['Benign','Malignant'])
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'MagQ-MoE Confusion Matrix\nThreshold={opt_threshold:.3f}')
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha='center',va='center')
plt.tight_layout(); plt.savefig(OUTPUT_DIR/'10_confusion_matrix.png', dpi=300); plt.show(); plt.close()

# Mean router weights by magnification
plot_df = routing_df.set_index('Magnification')[[f'Mean_Gate_{q}Q' for q in EXPERT_QUBITS]]
ax = plot_df.plot(kind='bar', figsize=(8,5))
ax.set_ylabel('Mean Router Weight'); ax.set_xlabel('Magnification')
ax.set_title('Quantum Expert Routing by Histopathology Magnification')
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUTPUT_DIR/'11_router_weights_by_magnification.png', dpi=300); plt.show(); plt.close()

# Hard expert selection percentages
plot_df2 = routing_df.set_index('Magnification')[[f'Hard_Select_{q}Q_Pct' for q in EXPERT_QUBITS]]
ax = plot_df2.plot(kind='bar', figsize=(8,5))
ax.set_ylabel('Hard Selection (%)'); ax.set_xlabel('Magnification')
ax.set_title('Preferred Quantum Expert by Magnification')
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig(OUTPUT_DIR/'12_hard_expert_selection.png', dpi=300); plt.show(); plt.close()

# ================================================================
# 17. CONFIG + ZIP
# ================================================================
pd.DataFrame([{
    'Experiment':'MagQ-MoE', 'Dataset':'BreaKHis', 'Seed':SEED,
    'Backbone':'ImageNet-pretrained ResNet18', 'FineTune':'layer4',
    'Experts':'4Q,6Q,8Q', 'VQC_Depth':VQC_DEPTH,
    'Router_Input':'512-D CNN features + 4-D magnification one-hot',
    'Router_Hidden':ROUTER_HIDDEN, 'Router_Temperature':ROUTER_TEMPERATURE,
    'Balance_Lambda':BALANCE_LAMBDA, 'LR_Backbone':LR_BACKBONE,
    'LR_Experts_Router':LR_HEAD, 'Epochs_Max':EPOCHS, 'Patience':PATIENCE,
    'Split':'Patient-wise 70/15/15 strategy, seed 42'
}]).to_csv(OUTPUT_DIR/'13_experiment_configuration.csv', index=False)

zip_path = shutil.make_archive(
    '/kaggle/working/MagQ_MoE_BreaKHis_Results', 'zip', OUTPUT_DIR
)
print('\n' + '='*90)
print('MAGQ-MOE EXPERIMENT COMPLETE')
print('='*90)
print('Results:', OUTPUT_DIR)
print('ZIP    :', zip_path)
print('\nUpload MagQ_MoE_BreaKHis_Results.zip for result analysis and the next ablation.')
