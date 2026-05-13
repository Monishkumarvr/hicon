from processors.pouring_processor import PouringProcessor


REPORT_ROWS = [
    # heat, foundry_actual, clustered_meta, before_clustering_raw, median_s, p75_s, baseline
    ("HEAT_0447", 8, 11, 20, 6.0, 14.0, 11),
    ("HEAT_0448", 23, 23, 23, 5.0, 7.0, 23),
    ("HEAT_0450", 36, 26, 26, 4.0, 6.0, 26),
    ("HEAT_0452", 27, 18, 20, 4.0, 6.0, 18),
    ("HEAT_0457", 2, 4, 5, 3.0, 4.0, 4),
    ("HEAT_0458", 3, 2, 3, 3.0, 4.0, 2),
    ("HEAT_0459", 3, 2, 3, 3.0, 4.0, 2),
    ("HEAT_0460", 3, 2, 3, 3.0, 4.0, 2),
    ("HEAT_0461", 9, 10, 10, 6.0, 8.0, 10),
    ("HEAT_0462", 9, 11, 17, 6.0, 14.0, 11),
    ("HEAT_0463", 11, 22, 22, 6.0, 10.0, 11),
    ("HEAT_0464", 7, 6, 11, 6.0, 14.0, 6),
    ("HEAT_0465", 7, 7, 7, 5.0, 6.0, 7),
    ("HEAT_0466", 5, 5, 5, 4.0, 5.0, 5),
    ("HEAT_0468", 31, 22, 29, 6.0, 8.0, 22),
    ("HEAT_0469", 30, 24, 28, 6.0, 8.0, 24),
]


def _reduce(row):
    heat, actual, clustered, raw, median, p75, baseline = row
    official, diagnostics = PouringProcessor.compute_physical_mould_count(
        clustered_count=clustered,
        reactive_count=raw,
        predictive_count=None,
        baseline_cluster_count=baseline,
        rescue_cluster_count=clustered,
        duration_stats={"count": raw, "median": median, "p75": p75, "total": raw * median},
        mode="physical",
    )
    return heat, actual, official, diagnostics


def test_report_rows_hit_named_physical_count_targets():
    reduced = {heat: official for heat, _actual, official, _diag in map(_reduce, REPORT_ROWS)}

    assert reduced["HEAT_0450"] == 36
    assert reduced["HEAT_0452"] == 27
    assert reduced["HEAT_0463"] == 11
    assert reduced["HEAT_0468"] == 31
    assert reduced["HEAT_0469"] == 30

    assert reduced["HEAT_0448"] == 23
    assert reduced["HEAT_0465"] == 7
    assert reduced["HEAT_0466"] == 5


def test_report_total_stays_within_acceptance_band_and_keeps_diagnostics():
    rows = [_reduce(row) for row in REPORT_ROWS]
    actual_total = sum(actual for _heat, actual, _official, _diag in rows)
    clustered_total = sum(row[2] for row in REPORT_ROWS)
    before_clustering_total = 259
    official_total = sum(official for _heat, _actual, official, _diag in rows)

    assert actual_total == 214
    assert clustered_total == 195
    assert before_clustering_total == 259
    assert official_total == 216
    assert abs(official_total - actual_total) <= 2
    assert abs(official_total - actual_total) < abs(clustered_total - actual_total)
    assert abs(official_total - actual_total) < abs(before_clustering_total - actual_total)

    for heat, _actual, official, diagnostics in rows:
        assert diagnostics["official_physical_mould_count"] == official, heat
        assert diagnostics["reactive_mould_count"] == diagnostics["pour_action_count"], heat
        assert "clustered_mould_count" in diagnostics, heat
        assert diagnostics["reason"], heat


def test_legacy_mode_preserves_clustered_count():
    official, diagnostics = PouringProcessor.compute_physical_mould_count(
        clustered_count=18,
        reactive_count=20,
        duration_stats={"count": 20, "median": 4.0, "p75": 6.0, "total": 80.0},
        mode="legacy",
    )

    assert official == 18
    assert diagnostics["official_physical_mould_count"] == 18
    assert diagnostics["reason"] == "legacy_clustered"
