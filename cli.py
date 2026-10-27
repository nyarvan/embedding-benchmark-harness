"""Command line interface of the benchmark harness.

    python cli.py run --models bge_m3,openai_3_small --chunking fixed,recursive

Arguments are validated before the pipeline, and with it torch and beir, is
imported, so mistakes and --help are reported instantly.
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from harness.cache import run_files
from harness.config import CHUNK_STRATEGIES, BenchmarkConfig
from harness.models import available_models, get_spec

DEFAULTS = BenchmarkConfig()
RESULTS_DIR = Path('results')
NO_CHUNKING = 'none'


class CliError(Exception):
    """A misuse of the command, reported without a traceback."""


def csv_list(value: str) -> list[str]:
    """Split 'a, b' into ['a', 'b'], dropping duplicates, rejecting blanks."""
    items = [item.strip() for item in value.split(',')]
    if not all(items):
        raise argparse.ArgumentTypeError(f'empty item in list: {value!r}')
    return list(dict.fromkeys(items))


def model_list(value: str) -> list[str]:
    names = csv_list(value)
    unknown = [name for name in names if name not in available_models()]
    if unknown:
        raise argparse.ArgumentTypeError(
            f'unknown model(s) {", ".join(unknown)}; '
            f'available: {", ".join(available_models())}'
        )
    return names


def strategy_list(value: str) -> tuple[str, ...]:
    """Parse the chunking strategies; 'none' means whole documents only."""
    names = csv_list(value)
    if names == [NO_CHUNKING]:
        return ()
    unknown = [n for n in names if n not in CHUNK_STRATEGIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f'unknown chunking strategy {", ".join(unknown)}; use '
            f'{", ".join(CHUNK_STRATEGIES)} or {NO_CHUNKING}'
        )
    return tuple(names)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'expected an integer, got {value!r}'
        ) from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f'must be positive, got {number}')
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (cheap: no heavy imports)."""
    parser = argparse.ArgumentParser(
        prog='cli.py', description='Embedding benchmark harness.'
    )
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser(
        'run', help='embed, retrieve and evaluate, then save the metrics'
    )
    run.add_argument(
        '--models', type=model_list, required=True, metavar='LIST',
        help=f'comma-separated, from: {", ".join(available_models())}',
    )
    run.add_argument(
        '--chunking', type=strategy_list, default=DEFAULTS.strategies,
        metavar='LIST',
        help=f'comma-separated from {", ".join(CHUNK_STRATEGIES)}, or '
             f'"{NO_CHUNKING}" (default: {",".join(DEFAULTS.strategies)})',
    )
    run.add_argument('--dataset', default=DEFAULTS.dataset,
                     help='BEIR dataset name (default: %(default)s)')
    run.add_argument('--split', default=DEFAULTS.split,
                     help='queries split (default: %(default)s)')
    run.add_argument('--subset-size', type=positive_int,
                     default=DEFAULTS.subset_size,
                     help='documents kept from the corpus '
                          '(default: %(default)s)')
    run.add_argument('--chunk-size', type=positive_int,
                     default=DEFAULTS.chunk_size,
                     help='chunk length in characters (default: %(default)s)')
    run.add_argument('--top-k', type=positive_int, default=DEFAULTS.top_k,
                     help='chunks retrieved per query before aggregation '
                          '(default: %(default)s)')
    run.add_argument('--data-dir', default=str(DEFAULTS.data_dir),
                     help='embedding cache directory (default: %(default)s)')
    run.add_argument('--output', type=Path, metavar='PATH',
                     help='metrics file (default: '
                          f'{RESULTS_DIR}/<dataset>-<subset>_<models>.json)')
    run.add_argument('--overwrite', action='store_true',
                     help='replace an existing metrics file')
    run.add_argument('--yes', action='store_true',
                     help='confirm calls to paid APIs')
    return parser


def default_output(config: BenchmarkConfig, models: Sequence[str]) -> Path:
    """Metrics file name derived from the dataset, subset and models."""
    name = f'{config.dataset}-{config.subset_size}_{"-".join(models)}.json'
    return RESULTS_DIR / name


def paid_models_to_confirm(
        config: BenchmarkConfig, models: Sequence[str]
) -> list[str]:
    """Paid models that still have embeddings to compute.

    A model whose cache is complete costs nothing, so it needs no confirmation.
    """
    return [
        name for name in models
        if get_spec(name).paid and any(
            not path.exists() for path in run_files(
                config.dataset, config.subset_size, name, config.split,
                config.strategies, config.chunk_size, config.data_dir,
            )
        )
    ]


def run_command(args: argparse.Namespace) -> int:
    """Check the request, then run the benchmark and print the table."""
    config = BenchmarkConfig(
        dataset=args.dataset, subset_size=args.subset_size, split=args.split,
        chunk_size=args.chunk_size, strategies=args.chunking,
        top_k=args.top_k, data_dir=args.data_dir,
    )
    output = args.output or default_output(config, args.models)

    if output.exists() and not args.overwrite:
        raise CliError(
            f'{output} already exists; pass --overwrite or choose --output'
        )

    paid = paid_models_to_confirm(config, args.models)
    if paid and not args.yes:
        raise CliError(
            f'{", ".join(paid)} will call a paid API for embeddings that are '
            'not cached yet; pass --yes to confirm'
        )

    # Heavy imports (torch, beir) only after the checks above
    from harness import pipeline

    print(f'Running {", ".join(args.models)} on {config.dataset} '
          f'({config.subset_size} documents, {config.split} queries)')
    results = pipeline.run_benchmark(config, args.models, output)
    print()
    print(pipeline.format_table(results))
    print(f'\nMetrics saved to {output}')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return run_command(args)
    except (CliError, ValueError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
