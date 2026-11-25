"""A whole-band CRNN with CTC and optional Wells CNN initialization."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from indus.cnn import SignCNN

BLANK = 0
HORIZONTAL_STRIDE = 8


class SealCRNN(nn.Module):
    """Map an impression band to sign emissions in right-to-left reading order."""

    def __init__(self, classes):
        super().__init__()
        self.encoder = SignCNN(n_classes=classes).features
        # The last pool keeps horizontal resolution for CTC, with the same CNN weights.
        self.encoder[-1][-1] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.recurrent = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.classify = nn.Linear(256, classes + 1)  # reserve index 0 for CTC blank

    def load_sign_cnn(self, cnn):
        """Copy the image encoder and seed the sign logits with the crop classifier."""
        self.encoder.load_state_dict(cnn.features.state_dict())
        with torch.no_grad():
            self.classify.weight[1:].copy_(cnn.head[-1].weight)
            self.classify.bias[1:].copy_(cnn.head[-1].bias)
            self.classify.bias[BLANK] = -2.0

    def forward(self, image, widths):
        visual = self.encoder(image).mean(dim=2).transpose(1, 2)
        lengths = widths.cpu() // HORIZONTAL_STRIDE
        visual = reading_order(visual, lengths)
        packed = pack_padded_sequence(visual, lengths, batch_first=True, enforce_sorted=False)
        context, _ = self.recurrent(packed)
        context, _ = pad_packed_sequence(context, batch_first=True, total_length=visual.shape[1])
        emissions = self.classify(visual + context)
        return emissions, lengths


def reading_order(features, lengths):
    """Reverse valid frames per band, keeping batch padding outside the valid region."""
    time = torch.arange(features.shape[1], device=features.device)
    reverse = (lengths.to(features.device)[:, None] - 1 - time).clamp_min(0)
    return features.gather(1, reverse[:, :, None].expand_as(features))


def collapse_ctc(best_frames):
    """Collapse consecutive duplicate CTC emissions, then remove blanks."""
    out = []
    previous = BLANK
    for sign in best_frames:
        sign = int(sign)
        if sign != BLANK and sign != previous:
            out.append(sign)
        previous = sign
    return out


def required_frames(labels):
    """Minimum CTC frames, including blanks between identical neighbouring labels."""
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))


def edit_distance(reference, prediction):
    """Count the fewest insertions, deletions or substitutions between sign sequences."""
    previous = list(range(len(prediction) + 1))
    for i, truth in enumerate(reference, start=1):
        current = [i]
        for j, guess in enumerate(prediction, start=1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (truth != guess)))
        previous = current
    return previous[-1]


def sequence_metrics(references, predictions):
    """Micro sign error rate and seal-level exact and length accuracy."""
    values = list(zip(references, predictions, strict=True))
    n_signs = sum(len(ref) for ref, _ in values)
    if not values or not n_signs:
        raise ValueError("Evaluation needs at least one labelled sign")
    edits = sum(edit_distance(ref, guess) for ref, guess in values)
    exact = sum(list(ref) == list(guess) for ref, guess in values)
    counts = sum(len(ref) == len(guess) for ref, guess in values)
    return {"seals": len(values), "signs": n_signs, "sign_error_rate": edits / n_signs,
            "exact": exact / len(values), "count_accuracy": counts / len(values)}
