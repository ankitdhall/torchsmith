import json
import re
from pathlib import Path

from torchsmith.tokenizers.text_tokenizer import TextTokenizer


class StringTokenizer(TextTokenizer):
    def __init__(
        self,
        tokens: set[str],
        *,
        n_jobs: int = 1,
        batch_size: int = 1000,
        verbose: bool = False,
        token_counts: dict[str, int] | None = None,
        token_id_offset: int = 0,
    ) -> None:
        super().__init__(n_jobs=n_jobs, batch_size=batch_size, verbose=verbose)
        self._build_mappings(
            tokens=tokens, token_counts=token_counts, token_id_offset=token_id_offset
        )
        if self.verbose:
            print(f"Created a mapping for {len(self)} unique tokens: {self._str_to_id}")

    def _build_mappings(
        self,
        *,
        tokens: set[str],
        token_counts: dict[str, int] | None = None,
        token_ids: dict[str, int] | None = None,
        token_id_offset: int = 0,
    ) -> None:
        if token_ids is None:
            # TODO: name the function better
            tokens = tokens.union({self.BOS, self.EOS})
            # TODO: make them properties?
            self._id_to_str = {
                (idx + token_id_offset): token
                for idx, token in enumerate(sorted(tokens))
            }
            self._str_to_id = {char: idx for idx, char in self._id_to_str.items()}
        else:
            assert self.BOS in token_ids and self.EOS in token_ids
            self._str_to_id = token_ids
            self._id_to_str = {idx: char for char, idx in self._str_to_id.items()}

        if token_counts:
            self._str_to_count = token_counts
        else:
            self._str_to_count = {char: -1 for char in self._str_to_id.keys()}

        self._str_longest_first: list[str] = sorted(
            self._str_to_id.keys(), key=len, reverse=True
        )
        pattern = "|".join(map(re.escape, self._str_longest_first))
        self.regex = re.compile(pattern)

    def get_dir_to_save(self, path: Path) -> Path:
        return path / "tokenizer"

    def save(self, path: str | Path) -> None:
        path_to_save = self.get_dir_to_save(Path(path))
        path_to_save.mkdir(parents=True, exist_ok=False)
        self._save_v2(path_to_save)

    def load(self, path: str | Path) -> None:
        path_to_save = self.get_dir_to_save(Path(path))
        version = "1"
        try:
            with open(path_to_save / "info.json") as f:
                info = json.load(f)
                version = info["version"]
        except Exception as e:
            print(f"Could not load 'info.json': {e!s}. \nLoading as v1 ...")

        if version == "2":
            self._load_v2(path_to_save)
        else:
            self._load_v1(path_to_save)

    def _save_v2(self, path: Path) -> None:
        with open(path / "info.json", "w") as f:
            json.dump({"version": "2"}, f, indent=4)
        with open(path / "token_to_id.json", "w") as f:
            json.dump(self.token_to_id, f, indent=4)
        with open(path / "token_to_count.json", "w") as f:
            json.dump(self._str_to_count, f, indent=4)

    def _load_v2(self, path: Path) -> None:
        with open(path / "token_to_id.json") as f:
            token_to_id = json.load(f)
        with open(path / "token_to_count.json") as f:
            token_to_count = json.load(f)
        self._build_mappings(
            tokens=set(token_to_id.keys()),
            token_counts=token_to_count,
            token_ids=token_to_id,
        )

    def _save_v1(self, path: Path) -> None:
        with open(path / "token_to_id.json", "w") as f:
            json.dump(self.token_to_id, f, indent=4)

    def _load_v1(self, path: Path) -> None:
        with open(path / "token_to_id.json") as f:
            token_to_id = json.load(f)
        self._build_mappings(tokens=set(token_to_id.keys()))

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._str_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return self._id_to_str

    def split_text(self, x: str) -> list[str]:
        tokens = []
        while x:
            match = self.regex.match(x)
            if match:
                token = match.group(0)
                tokens.append(token)
                x = x[len(token) :]
            else:
                raise ValueError(f"Could not tokenize '{x}'.")

        return tokens


class WordTokenizer(StringTokenizer):
    def split_text(self, x: str) -> list[str]:
        x = x.replace(" ", "")
        tokens = []
        while x:
            match = self.regex.match(x)
            if match:
                token = match.group(0)
                tokens.append(token)
                x = x[len(token) :]
            else:
                raise ValueError(f"Could not tokenize '{x}'.")

        return tokens
