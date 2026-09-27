import json
from pathlib import Path

from every_eval_ever.adapters.terminal_bench_2 import adapter
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file


def _entry(**overrides):
    entry = {
        'rank': 1,
        'agent': 'Example Agent',
        'model': 'GPT-5',
        'date': '2026-01-01',
        'agent_org': 'Example Org',
        'model_org': 'OpenAI',
        'accuracy': 50.0,
        'stderr': 2.0,
    }
    entry.update(overrides)
    return entry


def test_normalized_entries_convert_and_validate(tmp_path: Path):
    bundles = adapter.make_logs([_entry()], retrieved_timestamp='1234567890.0')
    output_dir = tmp_path / 'data' / 'terminal-bench-2.0'
    paths = adapter.export(bundles, output_dir)

    assert len(paths) == 1
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


def test_the_metric_carries_a_join_key_that_is_not_plain_accuracy():
    """A trial-averaged resolution rate must not join to registry `accuracy`.

    The metric had no `metric_id` at all, so nothing tied these scores together
    across refreshes. The registry carries no Terminal-Bench metric, so the id is
    namespaced: it claims a stable join key within this source and no global
    identity, the same shape `mmlu_pro` uses.
    """
    bundles = adapter.make_logs([_entry()], retrieved_timestamp='1234567890.0')

    metric = bundles[0][0].evaluation_results[0].metric_config
    assert metric.metric_id == 'terminal-bench-2.0.accuracy'
    # The percent scale and its bounds are the leaderboard's own and unchanged;
    # only the missing id is being filled in here.
    assert metric.metric_unit == 'percent'
    assert (metric.min_score, metric.max_score) == (0, 100)


def test_custom_leaderboard_url_is_recorded_as_source():
    leaderboard_url = 'https://example.com/terminal-bench-2'

    bundles = adapter.make_logs(
        [_entry()],
        retrieved_timestamp='1234567890.0',
        leaderboard_url=leaderboard_url,
    )

    eval_log = bundles[0][0]
    assert eval_log.evaluation_results[0].source_data.url == [leaderboard_url]


def test_rejected_entry_retains_source_provenance():
    bad_entry = _entry(model='')

    try:
        adapter.make_logs([bad_entry], retrieved_timestamp='1234567890.0')
    except SourceRecordsError as exc:
        assert exc.failures[0].source_ref == 'leaderboard row 0'
        assert exc.failures[0].source_record == bad_entry
        assert 'model' in exc.failures[0].reason
    else:
        raise AssertionError('expected invalid Terminal-Bench entry to fail')


FIXTURE = (
    Path(__file__).parent / 'data' / 'terminal_bench_2' / 'leaderboard.json'
)


def test_payload_rows_become_entries_and_hidden_rows_are_excluded():
    payload = json.loads(FIXTURE.read_text(encoding='utf-8'))

    result = adapter.parse_leaderboard_payload(payload)

    assert result.total_records == 3
    assert not result.failures
    assert len(result.exclusions) == 1
    assert "'hidden'" in result.exclusions[0].reason
    first, second = result.records
    assert first == {
        'rank': 1,
        'agent': 'NexAU-AHE',
        'model': 'GPT-5.5',
        'date': '2026-04-23',
        'agent_org': 'china-qijizhifeng',
        'model_org': 'OpenAI',
        'accuracy': 84.7191011236,
        'ci95_half_width': 2.0892351283,
    }
    # a row published "± N/A" carries no half-width
    assert second['ci95_half_width'] is None


def test_published_half_width_is_a_95_percent_interval(tmp_path: Path):
    payload = json.loads(FIXTURE.read_text(encoding='utf-8'))
    entries = adapter.parse_leaderboard_payload(payload).records

    bundles = adapter.make_logs(entries, retrieved_timestamp='1234567890.0')

    with_ci = bundles[0][0].evaluation_results[0].score_details
    interval = with_ci.uncertainty.confidence_interval
    assert with_ci.uncertainty.standard_error is None
    assert interval.confidence_level == 0.95
    assert interval.lower == 84.7191011236 - 2.0892351283
    assert interval.upper == 84.7191011236 + 2.0892351283
    assert bundles[1][0].evaluation_results[0].score_details.uncertainty is None
    for path in adapter.export(
        bundles, tmp_path / 'data' / 'terminal-bench-2.0'
    ):
        report = validate_file(path)
        assert report.valid, report.errors
