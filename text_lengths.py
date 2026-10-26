"""Text length statistics used to choose the chunk size."""
import numpy as np

from harness.loaders import load_beir_dataset


def load_text_lengths() -> np.ndarray:
    """Character length of every document text in the loaded subset."""
    documents, _, _ = load_beir_dataset(split='train')
    return np.array([len(doc['text']) for _, doc in documents])


def describe_lengths(lengths: np.ndarray) -> dict:
    """Median, 90th and 99th percentiles, min, mean and max of the lengths."""
    return {
        'Percentile': ', '.join(
            np.percentile(lengths, [50, 90, 99]).astype(str)
        ),
        'Min': int(np.min(lengths)),
        'Mean': float(np.mean(lengths)),
        'Max': int(np.max(lengths)),
    }


def estimate_fixed_chunks(
    lengths: np.ndarray, sizes: list[int]
) -> dict[str, float]:
    """For each chunk size: total fixed chunks and share of longer documents.

    Keys: 'chunks_<size>' (chunks at overlap 0) and 'share_gt_<size>' (share
    of documents that get split at all).
    """
    result = {}
    for size in sizes:
        result[f'chunks_{size}'] = float(np.ceil(lengths / size).sum())
        result[f'share_gt_{size}'] = float((lengths > size).mean())

    return result


if __name__ == '__main__':
    lengths = load_text_lengths()
    sizes = [500, 1000, 2000]

    print(describe_lengths(lengths))
    print(estimate_fixed_chunks(lengths, sizes))
