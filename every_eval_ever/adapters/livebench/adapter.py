"""Convert the LiveBench leaderboard releases to the EvalEval schema format.

Data source (https://livebench.ai), read from the site's own repository,
https://github.com/LiveBench/livebench.github.io:
- ``src/App.js`` lists the leaderboard releases (``YYYY-MM-DD``).
- ``public/table_<release>.csv`` holds each model's score per task.
- ``public/categories_<release>.json`` groups the tasks into categories.
- ``src/Table/modelLinks.js`` names each model's organization, display name
  and link; a model the site has no entry for has no stated organization.

The site shows a category's score as the mean of its tasks' scores and the
overall score as the mean of the category scores (``src/Table/Averaging.js``);
this adapter publishes the task scores and those two means, each naming its
aggregation level so a consumer takes one level, not several.

Usage:
    uv run python -m every_eval_ever.adapters.livebench.adapter
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.eval_card_registry import Registry
from every_eval_ever.helpers.fetch import fetch_text
from every_eval_ever.helpers.io import datastore_path_components

COLLECTION = 'livebench'
OUTPUT_DIR = f'data/{COLLECTION}'
SITE_URL = 'https://livebench.ai'
SITE_REPO = 'https://github.com/LiveBench/livebench.github.io'
RAW_BASE = (
    'https://raw.githubusercontent.com/LiveBench/livebench.github.io/main'
)
#: The benchmark's questions, per category.
DATASET_URL = 'https://huggingface.co/livebench'

#: Category labels the site uses that are abbreviations.
CATEGORY_SLUGS = {'IF': 'instruction_following'}

_RELEASE = re.compile(r"setSelectedDate\('(\d{4}-\d{2}-\d{2})'\)")


# -- source files -------------------------------------------------------------


def release_file(release: str, kind: str, suffix: str) -> str:
    return f'{RAW_BASE}/public/{kind}_{release.replace("-", "_")}.{suffix}'


def parse_releases(app_js: str) -> list[str]:
    """The release dates the site's release picker offers, oldest first."""
    releases = sorted(set(_RELEASE.findall(app_js)))
    if not releases:
        raise ValueError('no releases found in the site App.js')
    return releases


def parse_model_links(model_links_js: str) -> dict[str, dict[str, Any]]:
    """Return the site's model metadata, variants resolved like ``getModelInfo``.

    ``modelLinks.js`` is a JavaScript object literal: bare keys, double-quoted
    strings, trailing commas. It is rewritten to JSON rather than evaluated.
    """
    start = model_links_js.index('{')
    end = model_links_js.index('};', start) + 1
    body = model_links_js[start:end]
    # quote bare keys, only where a key can start (after { or ,), so ':' in
    # a quoted URL is never touched
    body = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', body)
    body = re.sub(r',(\s*[}\]])', r'\1', body)
    links = json.loads(body)
    lookup: dict[str, dict[str, Any]] = {}
    for base, info in links.items():
        lookup[base] = {k: v for k, v in info.items() if k != 'variants'}
        for variant in info.get('variants') or []:
            merged = dict(lookup[base])
            merged.update({k: v for k, v in variant.items() if k != 'rawName'})
            lookup[variant['rawName']] = merged
    return lookup


def parse_table(table_csv: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(table_csv)))
    if not rows or 'model' not in rows[0]:
        raise ValueError('release table has no model column')
    return rows


def category_slug(label: str) -> str:
    return CATEGORY_SLUGS.get(
        label, re.sub(r'\W+', '_', label.lower()).strip('_')
    )


# -- conversion ----------------------------------------------------------------


def _score(value: str | None) -> float | None:
    """A task cell as the site reads it (``parseFloat``); blank is absent."""
    try:
        number = float(value) if value not in (None, '') else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _result(
    release: str,
    name: str,
    score: float,
    level: str,
    description: str,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_result_id=f'{COLLECTION}/{release}/{name}',
        evaluation_name=f'{COLLECTION}/{name}',
        source_data=SourceDataUrl(
            dataset_name=COLLECTION,
            source_type='url',
            url=[DATASET_URL],
            additional_details={'release': release},
        ),
        evaluation_timestamp=release,
        metric_config=MetricConfig(
            evaluation_description=description,
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='percent',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=100,
        ),
        score_details=ScoreDetails(
            score=score,
            details={'aggregation_level': level},
        ),
    )


def convert_row(
    row: dict[str, str],
    release: str,
    categories: dict[str, list[str]],
    model_links: dict[str, dict[str, Any]],
    registry: Registry,
    retrieved_timestamp: str,
) -> tuple[EvaluationLog, str, str]:
    """One table row -> one EvaluationLog and its datastore directories."""
    model = (row.get('model') or '').strip()
    if not model:
        raise ValueError('row has no model name')
    info = model_links.get(model)
    organization = (info or {}).get('organization')
    if not organization:
        raise ValueError(
            f'{model}: the site names no organization for this model '
            '(no modelLinks.js entry)'
        )
    org = registry.org(organization)
    org_id = org.canonical_id or re.sub(r'\W+', '-', organization.lower())
    model_id = f'{org_id}/{model}'

    results = []
    category_scores = []
    for label, tasks in categories.items():
        slug = category_slug(label)
        task_scores = []
        for task in tasks:
            score = _score(row.get(task))
            if score is None:
                continue
            task_scores.append(score)
            results.append(
                _result(
                    release,
                    f'{slug}/{task}',
                    score,
                    'task',
                    f'LiveBench {release} task {task} ({label})',
                )
            )
        category_score = _mean(task_scores)
        category_scores.append(category_score)
        if category_score is not None:
            results.append(
                _result(
                    release,
                    slug,
                    category_score,
                    'category',
                    f'LiveBench {release} {label}: mean of its task scores',
                )
            )
    if category_scores and None not in category_scores:
        results.insert(
            0,
            _result(
                release,
                'overall',
                _mean(category_scores),
                'overall',
                f'LiveBench {release} overall: mean of the category scores',
            ),
        )
    if not results:
        raise ValueError(f'{model}: no task scores in the release table')

    details = {
        'livebench_organization': organization,
        'deployment_type': 'unknown',
        # the site marks open-weight models; its absence is not a claim
        'model_availability': (
            'open_weights' if info.get('openweight') is True else 'unknown'
        ),
        **org.provenance('developer'),
    }
    for key, field in (
        ('displayName', 'livebench_display_name'),
        ('url', 'livebench_model_url'),
        ('version', 'livebench_model_version'),
        ('reasoner', 'livebench_reasoner'),
        ('note', 'livebench_note'),
    ):
        if info.get(key) is not None:
            details[field] = str(info[key])

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'{COLLECTION}/{release}/{model}',
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=release,
        source_metadata=SourceMetadata(
            source_name=f'LiveBench {release}',
            source_type='documentation',
            source_organization_name='LiveBench',
            source_organization_url=SITE_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                'release': release,
                'source_repository': SITE_REPO,
                'source_file': release_file(release, 'table', 'csv'),
                'categories_file': release_file(release, 'categories', 'json'),
            },
        ),
        eval_library=EvalLibrary(
            name='livebench',
            version=release,
            additional_details={
                'github': 'https://github.com/LiveBench/LiveBench'
            },
        ),
        model_info=ModelInfo(
            name=model,
            id=model_id,
            developer=organization,
            additional_details={k: v for k, v in details.items() if v},
        ),
        evaluation_results=results,
    )
    _, developer_dir, model_dir = datastore_path_components(
        COLLECTION, model_id
    )
    return log, developer_dir, model_dir


def convert_release(
    release: str,
    table_csv: str,
    categories: dict[str, list[str]],
    model_links: dict[str, dict[str, Any]],
    registry: Registry,
    retrieved_timestamp: str,
) -> SourceConversionResult[tuple[EvaluationLog, str, str]]:
    rows = parse_table(table_csv)
    bundles = []
    failures: list[SourceRecordFailure] = []
    for index, row in enumerate(rows):
        try:
            bundles.append(
                convert_row(
                    row,
                    release,
                    categories,
                    model_links,
                    registry,
                    retrieved_timestamp,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{release_file(release, "table", "csv")} row {index + 1}',
                    reason=str(exc),
                    source_record=row,
                )
            )
    return SourceConversionResult(
        source_name=f'LiveBench {release}',
        total_records=len(rows),
        records=bundles,
        failures=failures,
    )


def export(
    bundles: list[tuple[EvaluationLog, str, str]], output_dir: Path
) -> list[Path]:
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=log,
            base_dir=output_dir,
            developer=developer,
            model_name=model_name,
        )
        for log, developer, model_name in bundles
    )


# -- CLI -------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and convert the LiveBench leaderboard releases.'
    )
    parser.add_argument(
        '--release',
        action='append',
        help='Convert only this release (YYYY-MM-DD); repeatable. '
        'Default: every release the site lists.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Collection directory (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--no-registry-resolve',
        action='store_true',
        help='Do not resolve organizations against the eval-card-registry.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    releases = args.release or parse_releases(
        fetch_text(f'{RAW_BASE}/src/App.js')
    )
    model_links = parse_model_links(
        fetch_text(f'{RAW_BASE}/src/Table/modelLinks.js')
    )
    registry = Registry(enabled=not args.no_registry_resolve)
    retrieved_timestamp = str(time.time())
    results = []
    for release in releases:
        try:
            table_csv = fetch_text(release_file(release, 'table', 'csv'))
            categories = json.loads(
                fetch_text(release_file(release, 'categories', 'json'))
            )
            result = convert_release(
                release,
                table_csv,
                categories,
                model_links,
                registry,
                retrieved_timestamp,
            )
        except Exception as exc:
            result = SourceConversionResult(
                source_name=f'LiveBench {release}',
                total_records=1,
                records=[],
                failures=[
                    SourceRecordFailure(
                        source_ref=release_file(release, 'table', 'csv'),
                        reason=f'could not read release {release}: {exc}',
                    )
                ],
            )
        paths = export(result.records, args.output_dir)
        print(
            f'LiveBench {release}: {len(paths)} of '
            f'{result.total_records} row(s) written'
        )
        results.append(result)

    combined = SourceConversionResult(
        source_name='LiveBench',
        total_records=sum(r.total_records for r in results),
        records=[b for r in results for b in r.records],
        failures=[f for r in results for f in r.failures],
    )
    if combined.failures:
        report = save_failure_report(
            combined, default_failure_report_path(args.output_dir)
        )
        print(f'{len(combined.failures)} row(s) not converted: {report}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
