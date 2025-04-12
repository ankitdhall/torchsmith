from itertools import chain
from typing import cast

import pandas as pd

from torchsmith.datahub.eminem import EminemDataset
from torchsmith.datahub.eminem import get_huggingface_dataset
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.tokenizers import StringTokenizer
from torchsmith.tokenizers.text_tokenizer import generate_samples_text
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
set_resource_limits(n_jobs=12, maximum_memory=26)

sequence_length = 256

train_df, test_df = get_huggingface_dataset()
tokenizer = StringTokenizer(
    tokens=set(
        chain.from_iterable(
            pd.concat([train_df, test_df])[EminemDataset.TEXT_COLUMN_NAME].tolist()
        )
    ),
    n_jobs=4,
)
print("\t\tNumber of tokens:", len(tokenizer))

experiment_dir = EXPERIMENT_DIR / "eminem"
print(f"Saving experiment to: {experiment_dir}")

train_config = TrainConfig(num_epochs=20, batch_size=256, num_workers=n_jobs)

train_dataset = EminemDataset(train_df, tokenizer, sequence_length=sequence_length)
test_dataset = EminemDataset(test_df, tokenizer, sequence_length=sequence_length)

data_handler = DataHandler(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    train_config=train_config,
)

transformer_config = GPT2Config(
    seq_len=sequence_length,
    dropout=0.05,
    num_stack=4,
)
transformer = GPT2Decoder.from_config(
    vocab_size=len(tokenizer),
    config=transformer_config,
)

trainer = TrainerAutoregression(
    data_handler=data_handler,
    tokenizer=tokenizer,
    train_config=train_config,
    transformer=transformer,
    loss_fn=cross_entropy,
    sequence_length=transformer_config.seq_len,
    generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
    show_plots=False,
    sample_every_n_epochs=10,
    save_dir=experiment_dir,
    save_every_n_epochs=10,
)
transformer, train_losses, test_losses, samples = trainer.train()

for sample in tokenizer.decode_batch(iter(samples.tolist())):
    print("------ START ------")
    print("".join(sample))
    print("------- END -------")
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
