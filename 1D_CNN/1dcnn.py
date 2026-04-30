import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from tap_model import FEATURE_COLUMNS, Tap1DCNN

# Path resolution
_SCRIPT_DIR = Path(__file__).parent.resolve()
_DEFAULT_CSV_DIR = _SCRIPT_DIR / "tap_data"
_DEFAULT_MODEL_OUTPUT = _SCRIPT_DIR / "model_output" / "best_1dcnn_tap.pt"

# Holds training configuration parameters
@dataclass
class Config:
    csv_path: str = str(_DEFAULT_CSV_DIR)
    window_size: int = 30
    stride: int = 1
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 1e-3
    test_size: float = 0.2
    random_state: int = 4524
    use_weighted_loss: bool = True
    split_by_session: bool = True
    group_k_folds: int = 1
    save_path: str = str(_DEFAULT_MODEL_OUTPUT)
    device: str = "cpu"

# Dataset for sliding window time series
class TapWindowDataset(Dataset):
    # Initializes dataset tensors
    def __init__(self, X_windows: np.ndarray, y_windows: np.ndarray):
        self.X = torch.tensor(X_windows, dtype=torch.float32)
        self.y = torch.tensor(y_windows, dtype=torch.float32)

    # Returns number of samples
    def __len__(self):
        return len(self.X)

    # Returns one sample formatted for Conv1d
    def __getitem__(self, idx):
        x = self.X[idx].transpose(0, 1)
        y = self.y[idx]
        return x, y

# Loads and validates single CSV data file, returns dataframe
def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = ["frame_idx", "timestamp", *FEATURE_COLUMNS, "label"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "session_id" not in df.columns:
        df["session_id"] = Path(csv_path).stem # add filename as "session_id"

    df["session_id"] = df["session_id"].astype(str)

    df = df.sort_values(by=["frame_idx"]).reset_index(drop=True)

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    invalid_labels = df.loc[~df["label"].isin([0, 1])]
    if not invalid_labels.empty:
        raise ValueError("Label column must contain only 0 or 1.")

    return df

# loads all csvs from multiple files if necessary
def load_and_validate_csvs(csv_paths):
    frames = []
    for path in csv_paths:
        df = load_and_validate_csv(path)
        if "session_id" not in df.columns:
            df["session_id"] = Path(path).stem
        df["session_id"] = df["session_id"].fillna(Path(path).stem).astype(str)
        frames.append(df)

    if not frames:
        raise ValueError("No CSV files were loaded.")

    return pd.concat(frames, ignore_index=True)

# Converts repeated 1s into single event points
def collapse_positive_runs_to_single_event(labels: np.ndarray) -> np.ndarray:
    collapsed = np.zeros_like(labels)
    i = 0
    n = len(labels)

    while i < n:
        if labels[i] == 1:
            start = i
            while i < n and labels[i] == 1:
                i += 1
            end = i - 1
            center = (start + end) // 2
            collapsed[center] = 1
        else:
            i += 1

    return collapsed

# Creates sliding windows and assigns labels
def create_sliding_windows(df, feature_columns, label_column, window_size, stride):
    X_windows = []
    y_windows = []
    window_session_ids = []

    raw_positive_frames = 0
    collapsed_tap_events = 0

    grouped = df.groupby("session_id", sort=False)
    for session_id, session_df in grouped:
        session_df = session_df.sort_values(by=["frame_idx"]).reset_index(drop=True)

        X = session_df[feature_columns].values.astype(np.float32)
        raw_y = session_df[label_column].values.astype(int)
        event_y = collapse_positive_runs_to_single_event(raw_y)

        raw_positive_frames += int(np.sum(raw_y == 1))
        collapsed_tap_events += int(np.sum(event_y == 1))

        if len(session_df) < window_size:
            continue

        for start in range(0, len(session_df) - window_size + 1, stride):
            end = start + window_size
            window_x = X[start:end]
            
            # Only label positive if the tap event is near the center of the window
            center = window_size // 2
            margin = 5  # frames of tolerance
            window_y = 1 if np.any(event_y[start + center - margin : start + center + margin] == 1) else 0

            X_windows.append(window_x)
            y_windows.append(window_y)
            window_session_ids.append(session_id)

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.float32)
    window_session_ids = np.array(window_session_ids)

    stats = {
        "num_rows": len(df),
        "raw_positive_frames": raw_positive_frames,
        "collapsed_tap_events": collapsed_tap_events,
        "num_sessions": int(df["session_id"].nunique()) if len(df) else 0,
        "sessions_with_windows": int(len(np.unique(window_session_ids))) if len(window_session_ids) else 0,
    }

    return X_windows, y_windows, window_session_ids, stats

# Standardizes feature values using training data
def standardize_windows(X_train, X_test):
    scaler = StandardScaler()

    num_train, win_size, num_features = X_train.shape
    num_test = X_test.shape[0]

    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, num_features)
    ).reshape(num_train, win_size, num_features)

    X_test_scaled = scaler.transform(
        X_test.reshape(-1, num_features)
    ).reshape(num_test, win_size, num_features)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


def split_windows(X, y, session_ids, cfg: Config):
    if cfg.split_by_session:
        unique_sessions = np.unique(session_ids)
        if len(unique_sessions) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=session_ids))

            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                print(
                    f"Split strategy: grouped by session "
                    f"(train sessions={len(np.unique(session_ids[train_idx]))}, "
                    f"test sessions={len(np.unique(session_ids[test_idx]))})"
                )
                return X[train_idx], X[test_idx], y_train, y_test

            print("Grouped split produced a single-class train/test set; falling back to stratified random split.")
        else:
            print("Not enough sessions for grouped split; falling back to stratified random split.")

    return train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

# Computes class imbalance weight
def compute_pos_weight(y_train, device):
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)

    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)

    return torch.tensor(neg / pos, dtype=torch.float32, device=device)

# Trains model for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)

# Evaluates model performance
@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.7):
    model.eval()
    total_loss = 0.0

    preds = []
    probs = []
    targets = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        prob = torch.sigmoid(logits)
        pred = (prob >= threshold).float()

        preds.extend(pred.cpu().numpy())
        probs.extend(prob.cpu().numpy())
        targets.extend(y.cpu().numpy())

        total_loss += loss.item() * X.size(0)

    preds = np.array(preds)
    probs = np.array(probs)
    targets = np.array(targets)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "f1": f1_score(targets, preds, zero_division=0),
        "y_true": targets,
        "y_pred": preds,
        "y_prob": probs,
    }

# Prints training progress
def print_epoch_metrics(epoch, train_loss, val_loss, val):
    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val['accuracy']:.4f} | "
        f"Val Precision: {val['precision']:.4f} | "
        f"Val Recall: {val['recall']:.4f} | "
        f"Val F1: {val['f1']:.4f}"
    )


def find_best_threshold(y_true, y_prob, min_threshold=0.5, max_threshold=0.90, step=0.01, beta=0.8):
    thresholds = np.arange(min_threshold, max_threshold + 1e-12, step)

    best_threshold = min_threshold
    best_score = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.float32)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

        if (
            score > best_score
            or (np.isclose(score, best_score) and precision > best_precision)
            or (np.isclose(score, best_score) and np.isclose(precision, best_precision) and recall > best_recall)
        ):
            best_threshold = float(threshold)
            best_score = float(score)
            best_precision = float(precision)
            best_recall = float(recall)

    return best_threshold, best_score, best_precision, best_recall

# Saves the best model checkpoint
def save_checkpoint(model, scaler, feature_columns, cfg, path, best_threshold=0.7):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "window_size": cfg.window_size,
            "stride": cfg.stride,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "best_threshold": best_threshold,
        },
        path,
    )

# Prints basic dataset summary
def print_data_summary(stats, X, y):
    print(f"Loaded rows: {stats['num_rows']}")
    print(f"Recording sessions: {stats['num_sessions']}")
    print(f"Sessions with windows: {stats['sessions_with_windows']}")
    print(f"Raw positive frames: {stats['raw_positive_frames']}")
    print(f"Collapsed tap events: {stats['collapsed_tap_events']}")
    print(f"Created windows: {len(X)}")
    print(f"Window shape: {X.shape}")
    print(f"Positive windows: {int(np.sum(y == 1))}")
    print(f"Negative windows: {int(np.sum(y == 0))}")

# Builds dataloaders for training and testing
def build_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    train_loader = DataLoader(
        TapWindowDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TapWindowDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

# Resolves local CSV path(s)
def resolve_csv_path(user_path: str):
    path = Path(user_path)

    if path.exists() and path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {path}")
        return [str(path)]

    if path.exists() and path.is_dir():
        csv_files = sorted([str(p) for p in path.glob("*.csv") if p.is_file()])
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {user_path}")
        return csv_files

    raise FileNotFoundError(
        f"Could not find file or directory at: {user_path}"
    )

# Builds config from command line arguments
def build_config():
    default_cfg = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=default_cfg.csv_path)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--group_k_folds", type=int, default=1,
                        help="Number of GroupKFold splits by session_id (set >1 to enable CV)")
    parser.add_argument("--save_path", type=str, default=str(_DEFAULT_MODEL_OUTPUT))
    parser.set_defaults(split_by_session=True)
    parser.add_argument("--split_by_session", dest="split_by_session", action="store_true",
                        help="Split train/test by session_id groups (default)")
    parser.add_argument("--no_split_by_session", dest="split_by_session", action="store_false",
                        help="Disable grouped split and use stratified random split")

    args = parser.parse_args()

    return Config(
        csv_path=args.csv,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        test_size=args.test_size,
        random_state=default_cfg.random_state,
        use_weighted_loss=default_cfg.use_weighted_loss,
        split_by_session=args.split_by_session,
        group_k_folds=args.group_k_folds,
        save_path=args.save_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def load_window_data(cfg: Config):
    print(f"Using device: {cfg.device}")
    csv_files = resolve_csv_path(cfg.csv_path)
    print(f"CSV input: {cfg.csv_path}")
    print(f"Loaded CSV files: {len(csv_files)}")

    df = load_and_validate_csvs(csv_files)
    feature_columns = FEATURE_COLUMNS

    X, y, session_ids, stats = create_sliding_windows(
        df=df,
        feature_columns=feature_columns,
        label_column="label",
        window_size=cfg.window_size,
        stride=cfg.stride
    )

    print_data_summary(stats, X, y)

    if len(X) == 0:
        raise ValueError("No windows were created. Increase session length or reduce window_size.")

    if len(np.unique(y)) < 2:
        raise ValueError("Window labels contain only one class, so training cannot proceed.")

    return X, y, session_ids, feature_columns

# Prepares dataset and loaders
def prepare_data(cfg: Config):
    X, y, session_ids, feature_columns = load_window_data(cfg)

    X_train, X_test, y_train, y_test = split_windows(X, y, session_ids, cfg)

    X_train, X_test, scaler = standardize_windows(X_train, X_test)
    train_loader, test_loader = build_dataloaders(
        X_train, X_test, y_train, y_test, cfg.batch_size
    )

    return train_loader, test_loader, scaler, feature_columns, y_train


def run_group_kfold(cfg: Config):
    from sklearn.model_selection import GroupKFold

    X, y, session_ids, feature_columns = load_window_data(cfg)
    unique_sessions = np.unique(session_ids)

    if not cfg.split_by_session:
        raise ValueError("Group K-fold requires --split_by_session.")

    if cfg.group_k_folds < 2:
        raise ValueError("group_k_folds must be >= 2 for GroupKFold.")

    if len(unique_sessions) < cfg.group_k_folds:
        raise ValueError(
            f"Need at least {cfg.group_k_folds} unique sessions, but found {len(unique_sessions)}."
        )

    gkf = GroupKFold(n_splits=cfg.group_k_folds)
    fold_metrics = []
    original_save_path = cfg.save_path

    print(f"\nRunning GroupKFold with {cfg.group_k_folds} folds (session-wise split)")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=session_ids), start=1):
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Fold {fold_idx}: skipped (single-class train/test set)")
            continue

        print(
            f"\nFold {fold_idx}/{cfg.group_k_folds} | "
            f"train sessions={len(np.unique(session_ids[train_idx]))}, "
            f"test sessions={len(np.unique(session_ids[test_idx]))}"
        )

        X_train, X_test, scaler = standardize_windows(X[train_idx], X[test_idx])
        train_loader, test_loader = build_dataloaders(
            X_train, X_test, y_train, y_test, cfg.batch_size
        )

        model, criterion, optimizer = build_training_components(cfg, len(feature_columns), y_train)

        fold_save_path = str(Path(original_save_path).with_name(
            f"{Path(original_save_path).stem}_fold{fold_idx}{Path(original_save_path).suffix}"
        ))
        cfg.save_path = fold_save_path

        run_training(model, train_loader, test_loader, criterion, optimizer, scaler, feature_columns, cfg)

        ckpt = torch.load(cfg.save_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_threshold = float(ckpt.get("best_threshold", 0.7))

        final = evaluate(model, test_loader, criterion, cfg.device, threshold=best_threshold)
        fold_metrics.append(
            {
                "accuracy": final["accuracy"],
                "precision": final["precision"],
                "recall": final["recall"],
                "f1": final["f1"],
                "threshold": best_threshold,
            }
        )

        print(
            f"Fold {fold_idx} Final | Threshold: {best_threshold:.2f} | "
            f"Acc: {final['accuracy']:.4f} | Prec: {final['precision']:.4f} | "
            f"Rec: {final['recall']:.4f} | F1: {final['f1']:.4f}"
        )

    cfg.save_path = original_save_path

    if not fold_metrics:
        raise ValueError("No valid folds were evaluated. Check class balance per session.")

    print("\nGroupKFold Summary")
    for metric_name in ["accuracy", "precision", "recall", "f1", "threshold"]:
        values = np.array([m[metric_name] for m in fold_metrics], dtype=np.float32)
        print(f"{metric_name.capitalize():>9}: {values.mean():.4f} ± {values.std(ddof=0):.4f}")

# Builds model, loss, and optimizer
def build_training_components(cfg: Config, num_features: int, y_train):
    model = Tap1DCNN(num_features).to(cfg.device)

    if cfg.use_weighted_loss:
        pos_weight = compute_pos_weight(y_train, cfg.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted loss with pos_weight = {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using unweighted loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    return model, criterion, optimizer

# Trains the model across all epochs
def run_training(model, train_loader, test_loader, criterion, optimizer, scaler, feature_columns, cfg: Config):
    best_f05 = -1.0
    early_stopping_patience = 3
    min_delta = 1e-4
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val = evaluate(model, test_loader, criterion, cfg.device)
        best_threshold, val_f05, val_precision_tuned, val_recall_tuned = find_best_threshold(
            val["y_true"],
            val["y_prob"],
            min_threshold=0.5,
        )
        print_epoch_metrics(epoch, train_loss, val["loss"], val)
        print(
            f"  Threshold sweep [0.50-0.95] -> best={best_threshold:.2f} | "
            f"Val Precision: {val_precision_tuned:.4f} | "
            f"Val Recall: {val_recall_tuned:.4f} | "
            f"Val F0.5: {val_f05:.4f}"
        )

        if val_f05 > best_f05 + min_delta:
            best_f05 = val_f05
            epochs_without_improve = 0
            save_checkpoint(model, scaler, feature_columns, cfg, cfg.save_path, best_threshold=best_threshold)
            print(f"Saved best model to {cfg.save_path} (threshold={best_threshold:.2f})")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch:02d} (best Val F0.5: {best_f05:.4f})")
                break

# Prints final evaluation results
def print_final_results(model, test_loader, criterion, cfg: Config, threshold=0.7):
    final = evaluate(model, test_loader, criterion, cfg.device, threshold=threshold)

    print("\nFinal Results")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {final['accuracy']:.4f}")
    print(f"Precision: {final['precision']:.4f}")
    print(f"Recall:    {final['recall']:.4f}")
    print(f"F1:        {final['f1']:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(final["y_true"], final["y_pred"]))

    print("\nClassification Report")
    print(classification_report(final["y_true"], final["y_pred"], digits=4))

# Run the training pipeline
def main():
    cfg = build_config()

    if cfg.group_k_folds > 1:
        run_group_kfold(cfg)
        return

    train_loader, test_loader, scaler, feature_columns, y_train = prepare_data(cfg)
    model, criterion, optimizer = build_training_components(cfg, len(feature_columns), y_train)
    run_training(model, train_loader, test_loader, criterion, optimizer, scaler, feature_columns, cfg)

    # Load best checkpoint before final evaluation
    ckpt = torch.load(cfg.save_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_threshold = float(ckpt.get("best_threshold", 0.7))
    
    print_final_results(model, test_loader, criterion, cfg, threshold=best_threshold)

if __name__ == "__main__":
    main()
