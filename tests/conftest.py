# -*- coding: utf-8 -*-
"""
Conftest.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def data_path():
    return Path(__file__).parent / "data"
