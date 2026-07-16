from utils.metrics import REGISTRY, MetricsRegistry
from utils.perf import timed_section


def test_percentiles_known_values():
    reg = MetricsRegistry()
    for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        reg.record_latency("probe.x", v)
    snap = reg.snapshot()
    stats = snap["latency_ms"]["probe.x"]
    assert stats["count"] == 10
    assert stats["max"] == 10
    assert stats["p50"] == 5  # index round(0.5*9)=4 (banker's rounding) -> ordered[4]=5
    assert stats["p99"] == 10


def test_snapshot_resets_by_default():
    reg = MetricsRegistry()
    reg.record_latency("probe.y", 5.0)
    first = reg.snapshot()
    assert "probe.y" in first["latency_ms"]
    second = reg.snapshot()
    assert second["latency_ms"] == {}


def test_snapshot_can_preserve_latencies():
    reg = MetricsRegistry()
    reg.record_latency("probe.z", 3.0)
    reg.snapshot(reset_latencies=False)
    second = reg.snapshot()
    assert "probe.z" in second["latency_ms"]


def test_counters_accumulate():
    reg = MetricsRegistry()
    reg.increment("drops")
    reg.increment("drops", 4)
    snap = reg.snapshot()
    assert snap["counters"]["drops"] == 5


def test_gauge_reads_live_value_at_snapshot_time():
    reg = MetricsRegistry()
    state = {"depth": 0}
    reg.register_gauge("queue.depth", lambda: state["depth"])
    state["depth"] = 7
    snap = reg.snapshot()
    assert snap["gauges"]["queue.depth"] == 7


def test_gauge_exception_is_swallowed_as_none():
    reg = MetricsRegistry()

    def boom():
        raise RuntimeError("gauge failed")

    reg.register_gauge("broken", boom)
    snap = reg.snapshot()
    assert snap["gauges"]["broken"] is None


def test_empty_latency_series_omitted_from_percentiles():
    reg = MetricsRegistry()
    snap = reg.snapshot()
    assert snap["latency_ms"] == {}


def test_timed_section_records_into_global_registry():
    name = "test.timed_section.marker"
    REGISTRY.snapshot()  # drain any prior samples under this name
    with timed_section(name, threshold_ms=10_000):  # high threshold: no log spam
        pass
    snap = REGISTRY.snapshot()
    assert name in snap["latency_ms"]
    assert snap["latency_ms"][name]["count"] == 1
