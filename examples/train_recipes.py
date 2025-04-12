from collections import Counter
from pathlib import Path
from typing import cast

import huggingface_hub
import torch
from huggingface_hub import list_repo_files
from torch.utils.data import DataLoader

from torchsmith.datahub.recipes import MyIterableDataset
from torchsmith.datahub.recipes import RecipesDataset
from torchsmith.datahub.recipes import get_huggingface_dataset
from torchsmith.datahub.recipes import get_iterable_dataset
from torchsmith.datahub.recipes import tokenize_and_chunk
from torchsmith.models.gpt2 import GPT2Decoder
from torchsmith.tokenizers import BytePairEncodingBuilder
from torchsmith.tokenizers.byte_pair_encoding_tokenizer import BytePairEncodingTokenizer
from torchsmith.tokenizers.text_tokenizer import generate_samples_text
from torchsmith.training.config import GPT2Config
from torchsmith.training.config import TrainConfig
from torchsmith.training.data import DataHandler
from torchsmith.training.losses import cross_entropy
from torchsmith.training.trainer_autoregression import TrainerAutoregression
from torchsmith.utils.constants import EXPERIMENT_DIR
from torchsmith.utils.constants import RANDOM_STATE
from torchsmith.utils.constants import TOKENIZERS_DIR
from torchsmith.utils.dtypes import GenerateSamplesProtocol
from torchsmith.utils.plotting import plot_losses
from torchsmith.utils.pytorch import get_device
from torchsmith.utils.pyutils import set_resource_limits

device = get_device()

"""
Train dataset has 1932340 rows
Test dataset has 214908 rows

Dataset loaded!
Loading tokenizer from disk!
Len after tokenizing and chunking ... {train}
4192820
"""

sequence_length = 256
num_tokens_max = 1000  # 500
n_jobs = 12
"""
20% of len(train_df) = 0.2 * 1932340 = 386468
50% of len(train_df) = 0.5 * 1932340 = 966170
For both 20% and 50%, the first 220 tokens are the same.
A sample size of 20% seems to be good enough to create the vocabulary.

---
Looks like sequence length of 256 is a good choice.

test:
After tokenizing and before chunking ...
Number of samples: 214565
Mean sample length: 199.9894670612635 +- 118.16539302423152
After tokenizing and chunking ...
Number of samples: 193538
Mean sample length: 256.0 +- 0.0



train:
After tokenizing and before chunking ...
Number of samples: 1932683
Mean sample length: 200.40426702154465 +- 118.29389695214824

Number of samples: 1747064
Mean sample length: 256.0 +- 0.0

"""
tokenizer_batch_size = 386468 // n_jobs
tokenizer_on_disk = TOKENIZERS_DIR / f"recipes_tokenizer_{num_tokens_max}_lower"

set_resource_limits(n_jobs=n_jobs, maximum_memory=26)

train_config = TrainConfig(num_epochs=10, batch_size=256, num_workers=n_jobs)
print("Loading dataset!")
train_df, test_df = get_huggingface_dataset()
print("Dataset loaded!")
load_precomputed = True

if load_precomputed:
    print("Loading tokenizer from disk ...")
    subfolder = "recipes_tokenizer_lowercase_1000/1000/tokenizer"
    files_to_load = [
        Path(f).name
        for f in list_repo_files("ankitdhall/BPE_tokenizer_params")
        if f.startswith(subfolder)
    ]
    assert len(files_to_load) != 0
    path_to_state = Path("")
    for _filename in files_to_load:
        path_to_state = Path(
            huggingface_hub.hf_hub_download(
                "ankitdhall/BPE_tokenizer_params",
                filename=_filename,
                subfolder="recipes_tokenizer_lowercase_1000/1000/tokenizer",
            )
        ).parent.parent
    tokenizer = BytePairEncodingTokenizer(set(), occurrence_count=Counter())
    tokenizer.load(path_to_state)
    print(f"Tokenizer loaded from: {tokenizer_on_disk}")
else:
    print("Computing tokenizer ...")
    print(f"Num samples in 'train_df': {len(train_df)}")
    print(f"Num samples in 'test_df': {len(test_df)}")
    tokenizer_to_load = None
    tokenizer_df = train_df.sample(frac=0.2, replace=False, random_state=RANDOM_STATE)
    tokenizer = BytePairEncodingBuilder(
        (
            value
            for index, value in tokenizer_df[RecipesDataset.TEXT_COLUMN_NAME].items()
        ),
        verbose=False,
        num_tokens_max=num_tokens_max,
        n_jobs=n_jobs,
        batch_size=tokenizer_batch_size,
        save_interval_in_tokens=5,
        save_dir=tokenizer_on_disk,
        tokenizer_to_load_from=tokenizer_to_load,
    ).get_tokenizer()
    tokenizer.save(tokenizer_on_disk)
train_iter_ds, test_iter_ds = get_iterable_dataset(num_shards=train_config.num_workers)
test_ds = tokenize_and_chunk(
    test_iter_ds, tokenizer=tokenizer, sequence_length=sequence_length, verbose=False
)
train_ds = tokenize_and_chunk(
    train_iter_ds, tokenizer=tokenizer, sequence_length=sequence_length, verbose=False
)

train_dataset = MyIterableDataset(train_ds)
test_dataset = MyIterableDataset(test_ds)
print("\t\tNumber of tokens:", len(tokenizer))

experiment_dir = EXPERIMENT_DIR / "recipes"
print(f"Saving experiment to: {experiment_dir}")


def collate(samples: list[dict[str, list]]) -> torch.Tensor:
    if isinstance(samples[0]["input"], torch.Tensor):
        return torch.stack([sample["input"] for sample in samples])
    else:
        return torch.tensor([sample["input"] for sample in samples])


train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    # Set to False as iterable datasets cannot be shuffled.
    # https://discuss.pytorch.org/t/using-chaindataset-to-combine-iterabledataset/85236/4
    shuffle=False,
    num_workers=train_config.num_workers,
    pin_memory=True,
    collate_fn=collate,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=train_config.batch_size,
    shuffle=False,
    num_workers=train_config.num_workers,
    pin_memory=True,
    collate_fn=collate,
)
data_handler = DataHandler(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    train_config=train_config,
)

transformer_config = GPT2Config(seq_len=sequence_length)
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
    sample_every_n_epochs=None,
    save_dir=experiment_dir,
    save_every_n_epochs=5,
    train_dataset_len=4192820,
)
transformer, train_losses, test_losses, samples = trainer.train()

for sample in tokenizer.decode_batch(iter(samples.tolist())):
    print("------ START ------")
    print("".join(sample))
    print("------- END -------")
plot_losses(train_losses, test_losses=test_losses, save_dir=experiment_dir, show=True)
