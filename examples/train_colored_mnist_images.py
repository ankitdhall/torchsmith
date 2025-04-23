from typing import cast

from torchsmith.datahub.colored_mnist import ColoredMNISTDataset
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.tokenizers.mnist_tokenizer import VQVAEMNIST
from torchsmith.tokenizers.mnist_tokenizer import generate_samples_colored_mnist_image
from torchsmith.tokenizers.vqvae_tokenizer import VQVAEImageTokenizer
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.dtypes import GenerateSamplesProtocol
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

device = get_device()

n_jobs = 12

set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

tokenizer = VQVAEImageTokenizer(vqvae=VQVAEMNIST(), batch_size=10000)

train_config = TrainConfig(num_epochs=100, batch_size=512, num_workers=n_jobs)
transformer_config = GPT2Config(seq_len=ColoredMNISTDataset.SEQUENCE_LENGTH)
transformer = GPT2Decoder.from_config(
    vocab_size=len(tokenizer),
    config=transformer_config,
)

train_dataset = ColoredMNISTDataset(split="train", tokenizer=tokenizer)
test_dataset = ColoredMNISTDataset(split="test", tokenizer=tokenizer)

print("\t\tNumber of tokens:", len(tokenizer))

experiment_dir = EXPERIMENT_DIR / "colored_mnist_images"
print(f"Saving experiment to: {experiment_dir}")

data_handler = DataHandler(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_config=train_config,
)
trainer = TrainerAutoregression(
    data_handler=data_handler,
    tokenizer=tokenizer,
    train_config=train_config,
    transformer=transformer,
    loss_fn=cross_entropy,
    sequence_length=transformer_config.seq_len,
    generate_samples_fn=cast(
        GenerateSamplesProtocol, generate_samples_colored_mnist_image
    ),
    show_plots=False,
    sample_every_n_epochs=1,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
transformer, train_losses, test_losses, samples = trainer.train()
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
