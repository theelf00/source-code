import unittest

import torch

from utils.checkpoint import extract_state_dict


class TestCheckpointUtils(unittest.TestCase):
    def test_extract_from_rich_checkpoint(self):
        state = {"layer.weight": torch.randn(2, 2)}
        ckpt = {"model_state_dict": state, "epoch": 1}
        extracted = extract_state_dict(ckpt)
        self.assertIn("layer.weight", extracted)

    def test_extract_from_plain_state_dict(self):
        state = {"layer.bias": torch.randn(2)}
        extracted = extract_state_dict(state)
        self.assertIn("layer.bias", extracted)


if __name__ == "__main__":
    unittest.main()
