from __future__ import annotations

import time
from pathlib import Path

from conftest import make_project_card

from recall.db import reset as reset_db
from recall.indexer import build_index
from recall.search import search

PROJECTS = [
    ("prj-marl-inventory-2024", "Decentralized MARL Inventory Management",
     "A multi-agent reinforcement learning system for inventory management.",
     ["reinforcement-learning", "marl"], ["pytorch", "python"]),
    ("prj-trendscribe-2023", "TrendScribe",
     "An automated content pipeline for trend analysis.",
     ["nlp", "automation"], ["python", "sqlite"]),
    ("prj-forecasta-2022", "Forecasta Energy Forecasting",
     "Time-series forecasting for energy demand.",
     ["forecasting", "time-series"], ["python", "pandas"]),
    ("prj-hackathon-2021", "Weekend Hackathon Entry",
     "A quick prototype built in 24 hours.",
     ["hackathon"], ["javascript"]),
    ("prj-thesis-tool-2023", "Thesis Data Tool",
     "A tool for processing thesis experiment data.",
     ["research", "data"], ["python"]),
    ("prj-webapp-2020", "Client Web App",
     "A freelance web application for a small business client.",
     ["freelance", "webapp"], ["javascript", "react"]),
    ("prj-cli-tool-2022", "Internal CLI Tool",
     "A command line tool for internal automation.",
     ["cli", "automation"], ["python"]),
    ("prj-dashboard-2021", "Analytics Dashboard",
     "A dashboard for visualizing analytics data.",
     ["dashboard", "analytics"], ["javascript"]),
    ("prj-rl-experiment-2024", "RL Experiment Suite",
     "Experiments comparing reinforcement learning algorithms.",
     ["reinforcement-learning", "research"], ["python", "pytorch"]),
    ("prj-farsi-project-2023", "Farsi NLP Project",
     "پردازش زبان طبیعی برای متون فارسی.",
     ["nlp", "farsi"], ["python"]),
]


def _populate(vault_path: Path):
    for doc_id, title, summary, tags, tech in PROJECTS:
        make_project_card(vault_path, doc_id, title, summary=summary, tags=tags, tech=tech)


def test_search_finds_correct_card_by_keyword(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    start = time.perf_counter()
    hits = search(db_path, "reinforcement learning inventory")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200
    assert hits
    assert hits[0].doc_id == "prj-marl-inventory-2024"


def test_search_farsi_query_matches_farsi_card(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)

    hits = search(db_path, "زبان طبیعی فارسی")
    assert hits
    assert hits[0].doc_id == "prj-farsi-project-2023"


def test_reindex_all_reproduces_identical_results(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    build_index(vault_path, db_path)
    first = [h.doc_id for h in search(db_path, "python automation")]

    reset_db(db_path)
    build_index(vault_path, db_path)
    second = [h.doc_id for h in search(db_path, "python automation")]

    assert first == second


def test_index_is_idempotent(vault_path: Path):
    _populate(vault_path)
    db_path = vault_path / ".recall" / "index.db"
    stats1 = build_index(vault_path, db_path)
    assert stats1.added == len(PROJECTS)

    stats2 = build_index(vault_path, db_path)
    assert stats2.added == 0
    assert stats2.updated == 0
    assert stats2.unchanged == len(PROJECTS)
