"""Shared utilities for evaluation data adapters."""

from .developer import get_developer, get_model_id
from .fetch import FetchError, fetch_csv, fetch_json
from .io import generate_output_path, sanitize_filename, save_evaluation_log
from .schema import (
    SCHEMA_VERSION,
    make_evaluation_log,
    make_evaluation_result,
    make_metric_config,
    make_model_info,
    make_source_metadata,
)

__all__ = [
    # developer.py
    'get_developer',
    'get_model_id',
    # fetch.py
    'fetch_json',
    'fetch_csv',
    'FetchError',
    # io.py
    'save_evaluation_log',
    'generate_output_path',
    'sanitize_filename',
    # schema.py
    'SCHEMA_VERSION',
    'make_metric_config',
    'make_evaluation_result',
    'make_source_metadata',
    'make_model_info',
    'make_evaluation_log',
]
