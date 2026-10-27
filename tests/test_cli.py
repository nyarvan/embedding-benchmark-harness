"""Tests for the command line: parsing, refusals, exit codes, arguments."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from harness import pipeline
from harness.cache import run_files

ROOT = Path(__file__).resolve().parents[1]
PERFECT = {'P@5': 0.2, 'Recall@10': 1.0, 'NDCG@10': 1.0}


@pytest.fixture
def runs(monkeypatch):
    """Replace the pipeline; `runs.calls` records every benchmark request."""
    state = SimpleNamespace(calls=[], error=None)

    def fake_run(config, model_names, output_path):
        if state.error:
            raise state.error
        state.calls.append(SimpleNamespace(
            config=config, models=list(model_names), output=Path(output_path)
        ))
        return [(f'{model_names[0]}_classic', PERFECT)]

    monkeypatch.setattr(pipeline, 'run_benchmark', fake_run)
    return state


def run_cli(tmp_path, *args):
    """Run the CLI with an isolated cache and metrics file."""
    base = ['run', '--data-dir', str(tmp_path / 'data'),
            '--output', str(tmp_path / 'out.json')]
    return cli.main([*base, *args])


def parse(*args):
    return cli.build_parser().parse_args(['run', *args])


# --- parsing -------------------------------------------------------------

def test_defaults_come_from_the_benchmark_config():
    args = parse('--models', 'bge_m3')

    assert args.dataset == 'scifact'
    assert args.subset_size == 500
    assert args.split == 'train'
    assert args.chunk_size == 500
    assert args.top_k == 50
    assert args.chunking == ('fixed', 'recursive')
    assert args.overwrite is False
    assert args.yes is False


def test_models_list_is_trimmed_and_deduplicated_in_order():
    args = parse('--models', 'openai_3_small, bge_m3 ,openai_3_small')

    assert args.models == ['openai_3_small', 'bge_m3']


def test_chunking_none_means_whole_documents_only():
    assert parse('--models', 'bge_m3', '--chunking', 'none').chunking == ()


def test_chunking_accepts_a_single_strategy():
    args = parse('--models', 'bge_m3', '--chunking', 'recursive')

    assert args.chunking == ('recursive',)


@pytest.mark.parametrize('bad_args', [
    [],                                              # --models is required
    ['--models', 'nope'],
    ['--models', 'bge_m3,,openai_3_small'],
    ['--models', 'bge_m3', '--chunking', 'semantic'],
    ['--models', 'bge_m3', '--chunking', 'none,fixed'],
    ['--models', 'bge_m3', '--chunk-size', '0'],
    ['--models', 'bge_m3', '--chunk-size', 'abc'],
    ['--models', 'bge_m3', '--top-k', '-5'],
    ['--models', 'bge_m3', '--subset-size', '1.5'],
])
def test_invalid_arguments_exit_with_usage_error(bad_args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse(*bad_args)

    assert exit_info.value.code == 2
    assert 'error' in capsys.readouterr().err


def test_a_command_is_required():
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2


def test_unknown_model_error_lists_the_available_models(capsys):
    with pytest.raises(SystemExit):
        parse('--models', 'nope')

    err = capsys.readouterr().err
    assert all(name in err for name in ('bge_m3', 'e5_large_v2'))


# --- running -------------------------------------------------------------

def test_arguments_reach_the_pipeline(runs, tmp_path):
    code = run_cli(
        tmp_path, '--models', 'bge_m3,e5_large_v2', '--chunking', 'recursive',
        '--dataset', 'nfcorpus', '--split', 'test', '--subset-size', '200',
        '--chunk-size', '300', '--top-k', '80',
    )

    call = runs.calls[0]
    assert code == 0
    assert call.models == ['bge_m3', 'e5_large_v2']
    assert call.output == tmp_path / 'out.json'
    assert call.config.dataset == 'nfcorpus'
    assert call.config.split == 'test'
    assert call.config.subset_size == 200
    assert call.config.chunk_size == 300
    assert call.config.top_k == 80
    assert call.config.strategies == ('recursive',)
    assert str(call.config.data_dir) == str(tmp_path / 'data')


def test_none_chunking_reaches_the_pipeline_as_no_strategies(runs, tmp_path):
    run_cli(tmp_path, '--models', 'bge_m3', '--chunking', 'none')

    assert runs.calls[0].config.strategies == ()


def test_results_table_and_output_path_are_printed(runs, tmp_path, capsys):
    run_cli(tmp_path, '--models', 'bge_m3')

    out = capsys.readouterr().out
    assert 'bge_m3_classic' in out
    assert 'NDCG@10' in out
    assert str(tmp_path / 'out.json') in out


def test_default_output_name_is_built_from_dataset_subset_and_models(
        runs, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    cli.main(['run', '--models', 'bge_m3,e5_large_v2',
              '--data-dir', str(tmp_path / 'data')])

    assert runs.calls[0].output == Path(
        'results/scifact-500_bge_m3-e5_large_v2.json'
    )


# --- refusals ------------------------------------------------------------

def test_existing_metrics_file_is_not_overwritten_by_default(
        runs, tmp_path, capsys
):
    (tmp_path / 'out.json').write_text('[]')

    code = run_cli(tmp_path, '--models', 'bge_m3')

    assert code == 1
    assert 'already exists' in capsys.readouterr().err
    assert runs.calls == []
    assert (tmp_path / 'out.json').read_text() == '[]'


def test_overwrite_flag_allows_replacing_the_metrics_file(runs, tmp_path):
    (tmp_path / 'out.json').write_text('[]')

    code = run_cli(tmp_path, '--models', 'bge_m3', '--overwrite')

    assert code == 0
    assert len(runs.calls) == 1


def test_paid_model_with_missing_embeddings_needs_confirmation(
        runs, tmp_path, capsys
):
    code = run_cli(tmp_path, '--models', 'openai_3_small')

    assert code == 1
    assert '--yes' in capsys.readouterr().err
    assert runs.calls == []


def test_confirmation_only_names_the_paid_models(runs, tmp_path, capsys):
    run_cli(tmp_path, '--models', 'bge_m3,openai_3_small')

    err = capsys.readouterr().err
    assert 'openai_3_small' in err
    assert 'bge_m3' not in err


def test_yes_flag_confirms_a_paid_model(runs, tmp_path):
    code = run_cli(tmp_path, '--models', 'openai_3_small', '--yes')

    assert code == 0
    assert runs.calls[0].models == ['openai_3_small']


def test_paid_model_with_a_complete_cache_needs_no_confirmation(
        runs, tmp_path
):
    for path in run_files(
            'scifact', 500, 'openai_3_small', 'train',
            ('fixed', 'recursive'), 500, tmp_path / 'data'
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    code = run_cli(tmp_path, '--models', 'openai_3_small')

    assert code == 0


def test_paid_model_with_a_partial_cache_still_needs_confirmation(
        runs, tmp_path
):
    documents = run_files(
        'scifact', 500, 'openai_3_small', 'train', (), 500, tmp_path / 'data'
    )[0]
    documents.parent.mkdir(parents=True)
    documents.touch()

    code = run_cli(tmp_path, '--models', 'openai_3_small')

    assert code == 1


def test_free_models_never_need_confirmation(runs, tmp_path):
    assert run_cli(tmp_path, '--models', 'bge_m3,e5_large_v2') == 0


# --- errors --------------------------------------------------------------

def test_value_errors_from_the_pipeline_become_a_one_line_error(
        runs, tmp_path, capsys
):
    runs.error = ValueError('OPENAI_API_KEY is not set')

    code = run_cli(tmp_path, '--models', 'bge_m3')

    assert code == 1
    assert capsys.readouterr().err.strip() == (
        'error: OPENAI_API_KEY is not set'
    )


def test_invalid_dataset_name_is_reported_not_raised(runs, tmp_path, capsys):
    code = run_cli(
        tmp_path, '--models', 'openai_3_small', '--dataset', '../etc'
    )

    assert code == 1
    assert 'dataset' in capsys.readouterr().err


def test_unexpected_errors_are_not_hidden(runs, tmp_path):
    runs.error = RuntimeError('boom')

    with pytest.raises(RuntimeError):
        run_cli(tmp_path, '--models', 'bge_m3')


# --- start-up cost -------------------------------------------------------

def test_building_the_parser_does_not_import_heavy_libraries():
    code = (
        'import sys, cli; cli.build_parser(); '
        'assert "torch" not in sys.modules; '
        'assert "beir" not in sys.modules; '
        'assert "harness.pipeline" not in sys.modules'
    )

    result = subprocess.run(
        [sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_help_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, 'cli.py', 'run', '--help'], cwd=ROOT,
        capture_output=True, text=True,
    )

    assert result.returncode == 0
    assert '--models' in result.stdout
