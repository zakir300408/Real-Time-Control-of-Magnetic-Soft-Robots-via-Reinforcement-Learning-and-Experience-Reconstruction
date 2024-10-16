import torch
import torch.nn as nn

class SignalModel(nn.Module):
    def __init__(self, input_size):
        super(SignalModel, self).__init__()

        # Encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Decoder for next state prediction
        self.decoder_next_state = self._create_decoder(64 * input_size)  # 64 comes from the last Conv1d output channels

    def _create_decoder(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 3 phases + 3 amplitudes
            nn.Sigmoid()
        )

    def forward(self, state, deltas):
        # Combine state and deltas, reshape for encoder input
        combined_input = torch.cat([state, deltas], dim=1).unsqueeze(1)

        # Pass through encoder
        encoded_features = self.encoder(combined_input)

        # Flatten the encoded features to feed into the decoder
        flattened_output = encoded_features.view(encoded_features.size(0), -1)

        # Predict next state
        next_state = self.decoder_next_state(flattened_output)

        return next_state

    def focal_loss(self, pred, target, alpha=1, gamma=2):
        # Instantiate and compute Focal Loss
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
        return loss_fn(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)

        # Focal Loss
        pt = torch.exp(-bce_loss)  # pt is the probability that the prediction is correct
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss


        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss