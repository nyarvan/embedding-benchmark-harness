"""Text chunking strategies and chunk-to-document bookkeeping."""
from collections.abc import Callable
from dataclasses import dataclass

# Coarse to fine: paragraph, line, sentence, word
RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    """A piece of a document that keeps the link to its source document."""

    chunk_id: str  # '<document_id>_<chunk_index>'
    document_id: str
    text: str


class Chunking:
    """Split texts into chunks of at most `size` characters.

    Args:
        size: maximum chunk length in characters.
        overlap: characters shared by neighbouring chunks (fixed chunker
            only; the recursive chunker requires 0).
    """

    def __init__(self, size: int, overlap: int = 0):
        self.size = size
        self.overlap = overlap

    def fixed_chunker(self, text: str) -> list[str]:
        """Cut text into windows of `size` characters.

        With overlap=0 joining the chunks restores the original text. The last
        chunk may be shorter than `size`. Empty text gives an empty list.
        """
        self._validate()

        step = self.size - self.overlap
        result = []

        for start in range(0, len(text), step):
            result.append(text[start:start + self.size])
            # Stop once a window reaches the end, otherwise overlapping
            # windows would produce tails fully contained in the last chunk
            if start + self.size >= len(text):
                break

        return result

    def recursive_chunker(self, text: str) -> list[str]:
        """Split by paragraphs, then lines, sentences and words.

        Every chunk is at most `size` characters. Separators at chunk
        boundaries are dropped and repeated separators collapse, so joining
        the chunks does not restore the original text exactly.
        """
        self._validate()

        if self.overlap:
            raise NotImplementedError(
                'recursive_chunker supports overlap=0 only'
            )

        if not text:
            return []

        return self._split(text, RECURSIVE_SEPARATORS)

    @staticmethod
    def chunk_documents(
        documents: list[tuple[str, dict[str, str]]],
        splitter: Callable[[str], list[str]],
    ) -> list[Chunk]:
        """Chunk the 'text' of every (doc_id, doc) pair.

        Chunk ids are '<doc_id>_<index>'; qrels refer to documents, so each
        chunk keeps its `document_id`.
        """
        result = []
        for doc_id, doc in documents:
            chunks = splitter(doc['text'])

            for i, piece in enumerate(chunks):
                result.append(Chunk(f'{doc_id}_{i}', doc_id, piece))

        return result

    def _validate(self) -> None:
        if self.size <= 0 or self.overlap < 0 or self.overlap >= self.size:
            raise ValueError(
                'size must be > 0 and 0 <= overlap < size, '
                f'got size={self.size}, overlap={self.overlap}'
            )

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Split by the coarsest usable separator, recurse into long parts."""
        if len(text) <= self.size:
            return [text]

        if not separators:
            # No natural boundary left: cut by length
            return self.fixed_chunker(text)

        sep, rest = separators[0], separators[1:]

        if sep not in text:
            return self._split(text, rest)

        pieces: list[str] = []
        for part in text.split(sep):
            # Empty parts appear around repeated or trailing separators
            if part:
                pieces.extend(self._split(part, rest))

        return self._merge(pieces, sep)

    def _merge(self, pieces: list[str], sep: str) -> list[str]:
        """Greedily join neighbouring pieces while they fit into `size`."""
        merged: list[str] = []
        current = ''

        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + len(sep) + len(piece) <= self.size:
                current = current + sep + piece
            else:
                merged.append(current)
                current = piece

        if current:
            merged.append(current)

        return merged
