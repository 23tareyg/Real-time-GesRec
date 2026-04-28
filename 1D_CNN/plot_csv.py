"""
plot_csv.py

Plot truth label over frames, with optional predicted confidence overlay.
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    if 'frame_idx' not in df.columns:
        raise ValueError("CSV must contain 'frame_idx' column")

    truth_col = None
    if 'label' in df.columns:
        truth_col = 'label'
    elif 'gt_label' in df.columns:
        truth_col = 'gt_label'

    if truth_col is None:
        raise ValueError("CSV must contain 'label' or 'gt_label' column")

    os.makedirs(output_dir, exist_ok=True)

    # Plot truth as step graph and predicted prob as optional overlay.
    plt.figure()
    plt.step(df['frame_idx'], df[truth_col], where='post', label='Truth label (0/1)', linewidth=1.3)
    if 'prob' in df.columns:
        plt.plot(df['frame_idx'], df['prob'], label='Predicted prob', linewidth=1.5, alpha=0.9)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Frame Index')
    plt.ylabel('Value')
    plt.title('Truth Label with Predicted Confidence Overlay')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    output_path = os.path.join(output_dir, 'tap_vs_frame.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tap vs no tap from CSV")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--out', type=str, default='airtap/outputs', help='Output folder')

    args = parser.parse_args()

    plot_csv(args.csv, args.out)