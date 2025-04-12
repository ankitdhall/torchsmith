from pathlib import Path

import torch
import torch.nn as nn

from torchsmith.utils.pytorch import add_save_load


def assert_model_params_equal(model_before: nn.Module, model_after: nn.Module) -> None:
    """Asserts that all parameters for 2 PyTorch models are identical."""
    for (name_before, param_before), (name_after, param_after) in zip(
        model_before.state_dict().items(), model_after.state_dict().items()
    ):
        assert name_before == name_after, (
            f"Parameter names mismatch: '{name_before}' != '{name_after}'"
        )
        assert torch.allclose(param_before, param_after), (
            f"Parameter '{name_before}' changed after loading!"
        )


def test_add_save_load(tmp_path: Path) -> None:
    @add_save_load
    class LinearModel(nn.Module):
        def __init__(self, *, num_input: int, num_output: int) -> None:
            super().__init__()
            self.fc = nn.Linear(num_input, num_output)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = LinearModel(num_input=5, num_output=3)
    x = torch.ones(5)
    output_before = model(x)

    filename = "my_linear_model.pth"
    model.save_model(tmp_path / filename)

    model_loaded = LinearModel.load_model(tmp_path / filename)
    output_after = model_loaded(x)

    torch.testing.assert_close(output_after, output_before)
    assert_model_params_equal(model_before=model, model_after=model_loaded)
