"""Run the benchmark with the settings below (cli.py will replace this)."""
from harness.pipeline import BenchmarkConfig, format_table, run_benchmark

# Models to run; the choices are registered in harness/models.py
MODEL_NAMES = ('openai_3_small',)
RESULTS_PATH = 'results/scifact_500docs_openai.json'

if __name__ == '__main__':
    results = run_benchmark(
        BenchmarkConfig(top_k=200), MODEL_NAMES, RESULTS_PATH
    )
    print(format_table(results))
