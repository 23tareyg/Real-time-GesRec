import torch.nn as nn


FEATURE_COLUMNS = [
    "dist_raw",
    "dist_norm",
    "velocity",
    "accel",
    "hand_conf",
    "thumb_x_pos",
    "thumb_y_pos",
    "pointer_x_pos",
    "pointer_y_pos",
]


class Tap1DCNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_features, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)