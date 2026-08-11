import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "/home/oskla261/MVDRIVE/public/Trajectories/Trajectories_V2_Deep_2D"
ROC_PLOT_PATH = "/home/oskla261/MVDRIVE/public/Trajectories/classifier_roc_curve.png"
PREDICTION_LOG_PATH = "/home/oskla261/MVDRIVE/public/Trajectories/complete_prediction_ledger.txt"

BATCH_SIZE = 16   
INITIAL_LR = 0.001  
EPOCHS = 50       
K_FOLDS = 5 

# OPTIMIZATION: Temperature factor to stretch confidence scales at validation/inference time
# T < 1.0 expands compressed raw logits away from the ambiguous 50% line
CONFIDENCE_TEMPERATURE = 0.4  

# ==========================================
# 1. DATASET & COLLATOR
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        self.labels = []
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        for f in sorted(files):
            self.samples.append(os.path.join(folder_path, f))
            self.labels.append(1 if "ADHD" in f else 0)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        traj = np.load(self.samples[idx])
        file_path = self.samples[idx] 
        return torch.from_numpy(traj).float(), torch.tensor(self.labels[idx]).float(), file_path

def collate_fn(batch):
    trajectories, labels, paths = zip(*batch) 
    lengths = torch.tensor([len(t) for t in trajectories])
    padded_trajectories = pad_sequence(trajectories, batch_first=True)
    return padded_trajectories, torch.stack(labels), lengths, paths 

# ==========================================
# 2. OPTIMIZATION: BINARY FOCAL LOSS CLASS
# ==========================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        # Calculate the focal modulating factor based on target state
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce_loss
        return loss.mean()

# ==========================================
# 3. MODEL (Sigmoid Single-Output 1D-CNN)
# ==========================================
class ADHDSequenceClassifier(nn.Module):
    def __init__(self, input_dim=2, num_filters=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(2), 
            nn.Dropout(0.3),
            
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  
        )

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.conv_layers(x) 
        x = x.view(x.size(0), -1) 
        return self.fc(x)

# ==========================================
# 4. K-FOLD TRAINING LOOP
# ==========================================
def train_kfold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    try:
        dataset = TrajectoryDataset(DATA_DIR)
        print(f"📊 Dataset loaded: {len(dataset)} subjects found.")
    except FileNotFoundError as e:
        print(e)
        return

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    all_true_labels = []
    all_pred_probs = []
    
    with open(PREDICTION_LOG_PATH, 'w') as log_file:
        log_file.write("=========================================================\n")
        log_file.write("          QUALITY CONTROL: FULL PREDICTION LEDGER         \n")
        log_file.write("=========================================================\n\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- 🌀 Fold {fold+1}/{K_FOLDS} ---")
        
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model = ADHDSequenceClassifier().to(device)
        optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        # OPTIMIZATION: Swapped standard BCE for Focal Loss to emphasize hard examples
        criterion = FocalLossWithLogits(gamma=2.0)

        best_fold_acc = 0
        best_fold_probs = []
        best_fold_trues = []
        best_fold_records = [] 

        for epoch in range(EPOCHS):
            model.train()
            for trajs, labels, _, _ in train_loader:
                trajs, labels = trajs.to(device), labels.to(device).unsqueeze(1)
                logits = model(trajs)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct, total = 0, 0
            epoch_probs = []
            epoch_trues = []
            epoch_records = [] 

            with torch.no_grad():
                for trajs, labels, _, paths in val_loader: 
                    trajs_dev = trajs.to(device)
                    outputs = model(trajs_dev)
                    
                    # Compute un-scaled probabilities for global metrics (ROC/AUC calculations)
                    base_probs = torch.sigmoid(outputs).squeeze(1)
                    epoch_probs.extend(base_probs.cpu().numpy())
                    epoch_trues.extend(labels.numpy())
                    
                    # Hard classifications use the standard 0.5 center line
                    preds = (base_probs >= 0.5).long()
                    labels_long = labels.to(device).long()
                    
                    correct += (preds == labels_long).sum().item()
                    total += labels_long.size(0)

                    for i in range(len(labels)):
                        pred_val = preds[i].item()
                        true_val = labels_long[i].item()
                        sub_name = os.path.basename(paths[i])
                        guess_str = "ADHD" if pred_val == 1 else "Control"
                        true_str = "ADHD" if true_val == 1 else "Control"
                        
                        # OPTIMIZATION: Apply scaling factor to separate tightly clustered probabilities
                        scaled_logits = outputs[i] / CONFIDENCE_TEMPERATURE
                        scaled_prob = torch.sigmoid(scaled_logits).item()
                        
                        confidence = scaled_prob if pred_val == 1 else (1 - scaled_prob)
                        
                        if pred_val == true_val:
                            status = "✅ CORRECT"
                        else:
                            status = "❌ WRONG  "
                            
                        epoch_records.append(
                            f"{status} | {sub_name:<30} | Guessed: {guess_str:<7} | Actual: {true_str:<7} | Confidence: {confidence:.2%}"
                        )
            
            acc = correct / total
            if acc > best_fold_acc:
                best_fold_acc = acc
                best_fold_probs = epoch_probs
                best_fold_trues = epoch_trues
                best_fold_records = epoch_records 
            
            scheduler.step()
            
        print(f"✅ Best Val Acc for Fold {fold+1}: {best_fold_acc:.2%}")
        fold_results.append(best_fold_acc)
        
        all_true_labels.extend(best_fold_trues)
        all_pred_probs.extend(best_fold_probs)

        with open(PREDICTION_LOG_PATH, 'a') as log_file:
            log_file.write(f"--- FOLD {fold+1} LEDGER (Best Epoch Validation Accuracy: {best_fold_acc:.2%}) ---\n")
            for record in sorted(best_fold_records):
                log_file.write(f"  {record}\n")
            log_file.write("\n")

    # Metrics aggregation
    all_true_labels = np.array(all_true_labels)
    all_pred_probs = np.array(all_pred_probs)
    
    fpr, tpr, thresholds = roc_curve(all_true_labels, all_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    youden_j = tpr - fpr
    best_cutoff = thresholds[np.argmax(youden_j)]

    summary_str = (
        f"=========================================================\n"
        f"                    FINAL PERFORMANCE                     \n"
        f"=========================================================\n"
        f"🏆 FINAL CROSS-VALIDATION ACCURACY: {np.mean(fold_results):.2%} (+/- {np.std(fold_results):.2%})\n"
        f"📈 TOTAL AREA UNDER THE CURVE (AUC): {roc_auc:.4f}\n"
        f"🎯 OPTIMAL DECISION CUTOFF (YOUDEN): {best_cutoff:.4f}\n"
        f"=========================================================\n"
    )
    
    print("\n" + summary_str)
    with open(PREDICTION_LOG_PATH, 'a') as log_file:
        log_file.write(summary_str)

    # Export Plot
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline Guess (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Quality Control: Classifier ROC Curve (Focal Loss 1D-CNN)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(ROC_PLOT_PATH, dpi=150)
    print(f"🖼️ ROC Plot safely saved to: {ROC_PLOT_PATH}")

if __name__ == "__main__":
    train_kfold()