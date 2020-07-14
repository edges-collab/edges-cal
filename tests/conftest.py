# -*- coding: utf-8 -*-
"""
Conftest.
"""

import pytest

from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def data_path():
    """Path to test data."""
    return Path(__file__).parent / "data"
