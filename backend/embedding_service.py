from __future__ import annotations

from typing import Any

from .config import Settings


def effective_max_tokens(model) -> int:
    values = [value for value in (
        getattr(model, 'max_seq_length', None),
        getattr(getattr(model, 'tokenizer', None), 'model_max_length', None),
    ) if isinstance(value, int) and 0 < value < 100000]
    if not values:
        raise ValueError('Embedding model does not expose a usable maximum sequence length')
    return min(values)


def validated_inputs(model, texts: list[str], kind: str, model_name: str) -> list[str]:
    formatted = [embedding_input(text, kind, model_name) for text in texts]
    limit = effective_max_tokens(model)
    counts = [len(model.tokenizer.encode(value, add_special_tokens=True, truncation=False))
              for value in formatted]
    for count in counts:
        if count > limit:
            raise ValueError(f'Embedding input exceeds model limit: {count}>{limit}')
    return formatted


def embedding_input(text: str, kind: str, model_name: str) -> str:
    mode = __import__('os').getenv('EMBEDDING_PREFIX_MODE', 'auto').strip().lower()
    if mode == 'auto':
        mode = 'e5' if 'e5' in model_name.lower().replace('_', '-') else 'none'
    return f'{kind}: {text}' if mode == 'e5' else text


class EmbeddingService:
    """Owns the process-wide SentenceTransformer instance."""

    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.model_name = settings.embedding_model
        print(f'[startup] Loading embedding model: {self.model_name}')
        self.model = SentenceTransformer(self.model_name)

    def encode_query(self, question: str) -> list[float]:
        formatted = validated_inputs(self.model, [question], 'query', self.model_name)
        return self.model.encode(formatted[0], normalize_embeddings=True).tolist()

    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        """Encode only the retrieved passages used by query-time selection.

        These vectors are deliberately not written back to the index.  The
        document embeddings already in Qdrant remain the source of retrieval.
        """
        if not passages:
            return []
        formatted = validated_inputs(self.model, passages, 'passage', self.model_name)
        vectors = self.model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, 'get_embedding_dimension', None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        return int(getter())

