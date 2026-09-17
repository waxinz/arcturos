import sys
from pathlib import Path

# Make src/ importable without installing the package (skeleton keeps deps to
# what requirements.txt declares).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient

from arcturos.main import create_app


@pytest.fixture()
def client(tmp_path):
    """TestClient against a fresh per-test SQLite database."""
    app = create_app(tmp_path / "arcturos_test.db")
    return TestClient(app)
