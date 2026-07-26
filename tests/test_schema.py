from __future__ import annotations

import pytest
from pydantic import ValidationError

from recall.schema import parse_card


def _base_fm(**overrides):
    fm = {
        "id": "prj-example-2024",
        "type": "project",
        "title": "Example Project",
        "subtype": "personal",
        "status": "completed",
        "provenance": {"method": "manual"},
        "created": "2026-07-26",
        "updated": "2026-07-26",
    }
    fm.update(overrides)
    return fm


def test_valid_project_card_parses():
    card = parse_card(_base_fm())
    assert card.id == "prj-example-2024"
    assert card.type == "project"


def test_invalid_status_rejected():
    with pytest.raises(ValidationError):
        parse_card(_base_fm(status="bogus"))


def test_invalid_id_slug_rejected():
    with pytest.raises(ValidationError):
        parse_card(_base_fm(id="Not A Slug"))


def test_partial_dates_accepted():
    card = parse_card(_base_fm(started="2024", ended="2025-02"))
    assert card.started == "2024"
    assert card.ended == "2025-02"


def test_bad_partial_date_rejected():
    with pytest.raises(ValidationError):
        parse_card(_base_fm(started="not-a-date"))


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        parse_card(_base_fm(type="bogus-type"))
