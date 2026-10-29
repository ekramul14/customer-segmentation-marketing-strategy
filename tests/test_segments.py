"""Checks that segment names match the data and the API serves them."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SEGMENTS = "data/processed/segment_names.csv"
PROFILES = "data/processed/cluster_profiles.csv"


def load():
    return pd.read_csv(SEGMENTS), pd.read_csv(PROFILES)


def test_every_cluster_has_one_unique_name():
    seg, prof = load()
    assert sorted(seg["cluster"]) == sorted(prof["cluster"])
    assert seg["name"].is_unique


def test_big_spenders_really_spend_the_most():
    seg, prof = load()
    top = prof.loc[prof["PURCHASES"].idxmax(), "cluster"]
    assert seg.set_index("cluster").at[top, "name"] == "Big Spenders"


def test_shares_add_up():
    seg, _ = load()
    assert abs(seg["pct_customers"].sum() - 100) < 0.1
    assert abs(seg["pct_purchases"].sum() - 100) < 0.1


def test_api_serves_data_driven_names():
    from fastapi.testclient import TestClient
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.api import app

    seg, _ = load()
    body = TestClient(app).get("/clusters").json()
    assert {c["name"] for c in body} == set(seg["name"])
