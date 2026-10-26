"""OpenAI embeddings API behind the same `encode` interface as local models."""
import numpy as np
import openai
from dotenv import load_dotenv


class OpenAIEncoder:
    """Encode texts with the OpenAI embeddings API.

    Texts are sent in batches. The API key is taken from the OPENAI_API_KEY
    environment variable, loaded from a local .env file when one exists.
    Rate-limit and connection errors are retried by the client
    (`max_retries`); any other error is raised to the caller.
    """

    def __init__(
            self, model: str = 'text-embedding-3-small', batch_size: int = 100
    ):
        load_dotenv()
        self.client = openai.OpenAI(max_retries=5)
        self.model = model
        self.batch_size = batch_size

    def encode(self, inputs: list[str], **kwargs) -> np.ndarray:
        """Return a float32 array of shape (len(inputs), dim), in input order.

        `kwargs` (batch_size, show_progress_bar, ...) are accepted for
        compatibility with sentence-transformers and ignored.
        """
        vectors = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start:start + self.batch_size]
            response = self.client.embeddings.create(
                input=batch, model=self.model
            )
            # The API labels each item with its position in the batch; sort
            # by it so the output order never depends on the response order
            vectors.extend(
                item.embedding
                for item in sorted(response.data, key=lambda i: i.index)
            )

        if len(vectors) != len(inputs):
            raise ValueError(
                f'Expected {len(inputs)} embeddings, got {len(vectors)}'
            )

        return np.array(vectors, dtype=np.float32)
