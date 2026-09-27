import csv
import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.livebench import adapter
from every_eval_ever.helpers.eval_card_registry import Registry
from every_eval_ever.validate import validate_file

FIXTURES = Path(__file__).parent / 'data' / 'livebench'
RELEASE = '2026-01-08'


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding='utf-8')


@pytest.fixture
def converted():
    return adapter.convert_release(
        RELEASE,
        _read('table_2026_01_08.csv'),
        json.loads(_read('categories_2026_01_08.json')),
        adapter.parse_model_links(_read('modelLinks.js')),
        Registry(),
        '1234567890.0',
    )


def _scores(log):
    return {
        result.evaluation_name: result.score_details.score
        for result in log.evaluation_results
    }


def test_releases_come_from_the_site_release_picker():
    assert adapter.parse_releases(_read('App.js')) == [
        '2025-04-25',
        '2025-05-30',
        '2025-11-25',
        '2025-12-23',
        '2026-01-08',
        '2026-06-25',
    ]


def test_variants_inherit_their_base_entry():
    links = adapter.parse_model_links(_read('modelLinks.js'))

    variant = links['claude-opus-4-5-20251101-high-effort']
    assert variant['organization'] == 'Anthropic'
    assert variant['displayName'] == 'Claude 4.5 Opus High Effort'
    assert variant['url'] == 'https://www.anthropic.com/news/claude-opus-4-5'


def test_rows_become_task_category_and_overall_results(converted):
    log, developer, model = converted.records[0][0], *converted.records[0][1:]
    row = next(csv.DictReader(_read('table_2026_01_08.csv').splitlines()))
    categories = json.loads(_read('categories_2026_01_08.json'))
    scores = _scores(log)

    assert (
        log.evaluation_id
        == 'livebench/2026-01-08/qwen3-235b-a22b-instruct-2507'
    )
    assert log.model_info.id == 'alibaba/qwen3-235b-a22b-instruct-2507'
    assert (developer, model) == ('alibaba', 'qwen3-235b-a22b-instruct-2507')
    assert log.model_info.additional_details['model_availability'] == (
        'open_weights'
    )
    # task scores are the table's cells
    assert scores['livebench/reasoning/zebra_puzzle'] == float(
        row['zebra_puzzle']
    )
    # a category is the mean of its tasks, overall the mean of categories
    category_means = [
        sum(float(row[task]) for task in tasks) / len(tasks)
        for tasks in categories.values()
    ]
    assert scores['livebench/reasoning'] == pytest.approx(
        sum(float(row[t]) for t in categories['Reasoning']) / 4
    )
    assert scores['livebench/overall'] == pytest.approx(
        sum(category_means) / len(category_means)
    )
    assert 'livebench/instruction_following' in scores
    levels = {
        result.evaluation_name: result.score_details.details[
            'aggregation_level'
        ]
        for result in log.evaluation_results
    }
    assert levels['livebench/overall'] == 'overall'
    assert levels['livebench/reasoning'] == 'category'
    assert levels['livebench/reasoning/zebra_puzzle'] == 'task'


def test_a_blank_cell_is_absent_not_zero(converted):
    log = converted.records[1][0]
    scores = _scores(log)

    assert log.model_info.additional_details['livebench_display_name'] == (
        'Claude 4.5 Opus High Effort'
    )
    assert 'livebench/language/typos' not in scores
    # the category averages only the tasks that have a score
    row = list(csv.DictReader(_read('table_2026_01_08.csv').splitlines()))[1]
    assert scores['livebench/language'] == pytest.approx(
        (float(row['connections']) + float(row['plot_unscrambling'])) / 2
    )


def test_a_model_without_an_organization_is_a_failure(converted):
    assert converted.total_records == 3
    assert len(converted.records) == 2
    (failure,) = converted.failures
    assert 'zephyr-7b-beta' in failure.reason
    assert 'no organization' in failure.reason
    assert failure.source_record['model'] == 'zephyr-7b-beta'


def test_records_validate_at_their_datastore_path(converted, tmp_path):
    paths = adapter.export(
        converted.records, tmp_path / 'data' / adapter.COLLECTION
    )

    assert len(paths) == 2
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors
