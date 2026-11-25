"""Regression checks for CTC alignment and the seal reading direction."""

import pytest

torch = pytest.importorskip("torch")

from indus.crnn import SealCRNN, collapse_ctc, reading_order, required_frames, sequence_metrics


def test_ctc_repeated_signs_need_blank():
    assert collapse_ctc([0, 4, 4, 0, 4, 0, 9, 9]) == [4, 4, 9]
    assert required_frames([4, 4, 9]) == 4


def test_reverse_only_the_valid_band_frames():
    frames = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [99.0]]])
    actual = reading_order(frames, torch.tensor([3, 2]))
    assert actual[0, :, 0].tolist() == [3, 2, 1]
    assert actual[1, :2, 0].tolist() == [5, 4]


def test_crnn_ctc_accepts_variable_width_targets():
    torch.set_num_threads(2)
    model = SealCRNN(4)
    logits, lengths = model(torch.zeros(2, 1, 64, 128), torch.tensor([128, 80]))
    assert logits.shape == (2, 16, 5)
    assert lengths.tolist() == [16, 10]
    criterion = torch.nn.CTCLoss()
    loss = criterion(logits.log_softmax(-1).transpose(0, 1), torch.tensor([1, 1, 2, 3, 4]),
                     lengths, torch.tensor([3, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classify.weight.grad.isfinite().all()


def test_sign_error_rate_charges_insertions_deletions_and_substitutions():
    metrics = sequence_metrics([[1, 2], [3, 4], [5]], [[1, 2, 2], [3], [6]])
    assert metrics == {"seals": 3, "signs": 5, "sign_error_rate": 0.6, "exact": 0.0,
                       "count_accuracy": 1 / 3}
