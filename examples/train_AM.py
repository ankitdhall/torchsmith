from pathlib import Path
from typing import cast

import huggingface_hub
import pandas as pd

from torchsmith.datahub.arctic_monkeys import ArcticMonkeyDataset
from torchsmith.datahub.arctic_monkeys import get_huggingface_dataset
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.tokenizers import BytePairEncodingBuilder
from torchsmith.tokenizers import StringTokenizer
from torchsmith.tokenizers.text_tokenizer import generate_samples_text
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.constants import TOKENIZERS_DIR
from torchsmith.utils.dtypes import GenerateSamplesProtocol
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device

device = get_device()

sequence_length = 128
num_tokens_max = None  # 1000
tokenizer_on_disk = TOKENIZERS_DIR / f"AM_tokenizer_{num_tokens_max}"
train_df, test_df = get_huggingface_dataset()
load_precomputed = True

if load_precomputed:
    print("Loading tokenizer ...")
    path_to_state = Path(
        huggingface_hub.hf_hub_download(
            "ankitdhall/BPE_tokenizer_params",
            filename="token_to_id.json",
            subfolder="arctic_monkeys_tokenizer_None/tokenizer",
        )
    )
    tokenizer = StringTokenizer(set())
    tokenizer.load(path_to_state.parent.parent)
else:
    print("Computing tokenizer ...")
    tokenizer = BytePairEncodingBuilder(
        pd.concat([train_df, test_df])["text"].tolist(),
        verbose=False,
        num_tokens_max=num_tokens_max,
        n_jobs=1,
    ).get_tokenizer()
    tokenizer.save(tokenizer_on_disk)
print("\t\tNumber of tokens:", len(tokenizer))

experiment_dir = EXPERIMENT_DIR / "AM"
print(f"Saving experiment to: {experiment_dir}")

train_config = TrainConfig(num_epochs=80, batch_size=256, num_workers=12)

transformer_config = GPT2Config(seq_len=sequence_length)
transformer = GPT2Decoder.from_config(
    vocab_size=len(tokenizer),
    config=transformer_config,
)

train_dataset = ArcticMonkeyDataset(
    train_df, tokenizer, sequence_length=sequence_length
)
test_dataset = ArcticMonkeyDataset(test_df, tokenizer, sequence_length=sequence_length)

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
    generate_samples_fn=cast(GenerateSamplesProtocol, generate_samples_text),
    show_plots=False,
    sample_every_n_epochs=None,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
)
transformer, train_losses, test_losses, samples = trainer.train()

print(tokenizer.decode_batch(samples.tolist()))
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
