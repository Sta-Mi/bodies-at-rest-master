import unittest

import torch

from model import MattressFusionModel


class MattressFusionModelTest(unittest.TestCase):
    def test_full_and_missing_modality_forward(self):
        model = MattressFusionModel(16, 12, identity_dim=8, hidden_dim=32)
        features = {
            "sleepfm": torch.randn(2, 3, 16),
            "pressure": torch.randn(2, 3, 1, 64, 27),
            "pose": torch.randn(2, 3, 12),
            "identity": torch.randn(2, 3, 8),
        }
        self.assertEqual(tuple(model(features).shape), (2, 1))
        self.assertEqual(tuple(model({"sleepfm": features["sleepfm"]}).shape), (2, 1))

    def test_rejects_all_masked_sample(self):
        model = MattressFusionModel(16, 12, identity_dim=8, hidden_dim=32)
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            model({"sleepfm": torch.randn(1, 2, 16)}, {"sleepfm": torch.zeros(1, 2)})


if __name__ == "__main__":
    unittest.main()
