from pathlib import Path

import numpy as np
import torch
import torch.nn

from torchsmith.utils.dtypes import ModuleT
from torchsmith.utils.pyutils import get_arguments


def get_device(verbose: bool = False) -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if verbose:
        print(f"Using {device} device")
    return device


def add_save_load(cls: type[ModuleT]) -> type[ModuleT]:
    """Class decorator to add save and load methods for PyTorch models."""

    # Store original `__init__` to call later.
    original_init = cls.__init__

    def new_init(self: ModuleT, *args, **kwargs) -> None:
        """Wrapped ``__init__`` method to capture arguments."""
        # Call original `__init__`.
        original_init(self, *args, **kwargs)
        self.init_args = get_arguments(original_init, self, *args, **kwargs)

    # Replace `__init__` with new_init.
    cls.__init__ = new_init

    def save_model(self: ModuleT, filepath: Path | str) -> None:
        """Saves model state along with constructor arguments."""
        model_data = {
            "state_dict": self.state_dict(),
            "init_args": self.init_args,
        }
        torch.save(model_data, filepath)
        print(
            f"'{self.__class__.__name__}' saved to '{filepath}' with args:\n"
            f"{self.init_args}"
        )

    @classmethod  # type: ignore
    def load_model(cls: type[ModuleT], filepath: Path | str) -> ModuleT:
        """Loads a model from a file and returns an instance of the class."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No such file: '{filepath}'.")

        model_data = torch.load(filepath, map_location=get_device())
        init_args = model_data.get("init_args", {})

        model = cls(**init_args)  # Instantiate model with saved arguments
        model.load_state_dict(model_data["state_dict"])
        model.eval()  # Set model to evaluation mode
        print(f"'{cls.__name__}' loaded from '{filepath}' with args:\n{init_args}")
        return model

    # Attach methods to class
    cls.save_model = save_model
    cls.load_model = load_model

    return cls


def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("Trainable parameters:")
    for i, param in enumerate(trainable_params, 1):
        print(f"{i}. Shape: {param.shape}, Number of elements: {param.numel()}")

    total_params = sum(p.numel() for p in trainable_params)
    print(f"\nTotal trainable parameters: {total_params}")


def print_named_parameters(model: torch.nn.Module) -> None:
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"Layer: {name} | Parameters: {num_params}")
    total_params = sum([np.prod(p.shape) for p in model.parameters()])
    print(f"Total parameters {model.__class__.__name__}: {total_params / 1e6:.3f}M")
