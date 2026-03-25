from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

from every_eval_ever import cli


def _make_inspect_args(log_path: Path, output_dir: Path) -> Namespace:
    return Namespace(
        log_path=str(log_path),
        output_dir=str(output_dir),
        source_organization_name='TestOrg',
        evaluator_relationship='third_party',
        source_organization_url=None,
        source_organization_logo_url=None,
        eval_library_name='inspect',
        eval_library_version='unknown',
    )


def test_convert_inspect_uses_detailed_results_uuid_for_aggregate_file(
    tmp_path, monkeypatch
):
    log_path = tmp_path / 'inspect_log.json'
    log_path.write_text('{}', encoding='utf-8')

    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    fake_log = SimpleNamespace(
        detailed_evaluation_results=SimpleNamespace(
            file_path=(
                f'/tmp/some_dataset/some_model/{uuid_value}_samples.jsonl'
            )
        )
    )

    fake_module = ModuleType('every_eval_ever.converters.inspect.adapter')

    class FakeInspectAdapter:
        def transform_from_file(self, *_args, **_kwargs):
            return fake_log

        def transform_from_directory(self, *_args, **_kwargs):
            return [fake_log]

    fake_module.InspectAIAdapter = FakeInspectAdapter
    monkeypatch.setitem(
        sys.modules, 'every_eval_ever.converters.inspect.adapter', fake_module
    )

    captured_eval_uuids: list[str | None] = []

    def fake_write_log(_log, _base_output, eval_uuid=None):
        captured_eval_uuids.append(eval_uuid)
        return Path('/tmp/fake_aggregate.json')

    monkeypatch.setattr(cli, '_write_log', fake_write_log)

    rc = cli._cmd_convert_inspect(_make_inspect_args(log_path, tmp_path))

    assert rc == 0
    assert captured_eval_uuids == [uuid_value]


def test_extract_file_uuid_from_detailed_results_parses_uuid_filename():
    uuid_value = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'
    fake_log = SimpleNamespace(
        detailed_evaluation_results=SimpleNamespace(
            file_path=f'/tmp/some_dataset/some_model/{uuid_value}_samples.JSONL'
        )
    )

    assert (
        cli._extract_file_uuid_from_detailed_results(fake_log) == uuid_value
    )
