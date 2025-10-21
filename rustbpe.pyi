# nanochat/rustbpe.pyi

from collections.abc import Iterable

class Tokenizer:
    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int,
        buffer_size: int = ...,
        pattern: str | None = ...,
    ) -> None: ...
    def get_mergeable_ranks(self) -> list[tuple[bytes, int]]: ...
    def encode(
        self,
        text: str | list[str],
        *,
        prepend: int | str | None = None,
        append: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]: ...
    def encode_special(self, token: str) -> int: ...
    def get_pattern(self) -> str: ...
