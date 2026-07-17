"""P0 regression tests (hicon-75h): pouring nvinfer interval>0 must not break
session capture.

With interval=1, tracker-propagated objects on non-inference frames carry
obj_meta.confidence == -0.1; the raw conf gates emptied mouth/trolley lists every
other frame and the session accumulator reset — zero pours were recorded for 16h.
"""

from pathlib import Path

import processors.pouring_processor as pp_mod
from processors.pouring_processor import PouringProcessor


class DummyDB:
    def insert_pouring_event(self, **kwargs):
        return 1

    def update_pouring_end(self, **kwargs):
        return None

    def insert_heat_cycle(self, **kwargs):
        return 1


class DummyConfig:
    CUSTOMER_ID = "C1"
    LOCATION = "Loc"
    CAMERA_ID_STREAM_0 = "Cam-0"
    MOUTH_CONFIDENCE = 0.4
    TROLLEY_CONFIDENCE = 0.25
    MOULD_GIE_ENABLED = True
    MOULD_GIE_UNIQUE_ID = 4
    MOULD_TRACKER_CLASS_ID = 2
    MOULD_MIN_AREA_PX = 400
    MOULD_COUNT_MODE = "shadow"
    STREAM_0_TRACKER_MAX_TARGETS = 64
    SESSION_START_DURATION = 1.0
    SESSION_END_DURATION = 1.5
    POUR_REF_WIDTH = 1920
    POUR_REF_HEIGHT = 1080
    POUR_PROBE_BELOW_PX = 30
    POUR_PROBE_OFFSETS = [(0, 0), (12, 0)]
    POUR_PROBE_RADIUS_PX = 8
    POUR_BRIGHTNESS_START = 205
    POUR_BRIGHTNESS_END = 160
    POUR_START_DURATION = 0.20
    POUR_END_DURATION = 0.80
    POUR_MIN_DURATION = 2.0
    MOULD_DISPLACEMENT_THRESHOLD = 0.25
    MOULD_SUSTAINED_DURATION = 0.30
    CLUSTER_R_CLUSTER = 0.08
    CLUSTER_R_MERGE = 0.07
    CLUSTER_BACKTRACK_CID_GUARD = 5
    MOULD_SWITCH_MIN_POUR_S = 2.0
    MIN_CLUSTER_POUR_S = 1.5
    EDGE_EXPAND_PX = 180
    MOUTH_MISSING_TOL_S = 0.6
    MOUTH_HOLD_S = 0.4
    PHANTOM_TROLLEY_TIMEOUT_S = 5.0
    POURING_CYCLE_TIMEOUT_S = 300.0
    ENABLE_INFERENCE_VIDEO = False
    VIDEO_DIR = Path("/tmp")
    POUR_NVINFER_INTERVAL = 1  # the regression trigger
    MOULD_DIAG_CSV = False


def _make_proc(tmp_path, interval=1):
    cfg = DummyConfig()
    cfg.POUR_NVINFER_INTERVAL = interval
    return PouringProcessor(
        db_manager=DummyDB(),
        config=cfg,
        screenshot_dir=str(tmp_path),
        heat_cycle_manager=None,
    )


class _FakeRect:
    def __init__(self, x1, y1, x2, y2):
        self.left = x1
        self.top = y1
        self.width = x2 - x1
        self.height = y2 - y1


class _FakeText:
    display_text = ""


class _FakeObj:
    def __init__(self, class_id, conf, bbox, track_id, gie_id=1):
        self.class_id = class_id
        self.confidence = conf
        self.rect_params = _FakeRect(*bbox)
        self.text_params = _FakeText()
        self.object_id = track_id
        self.unique_component_id = gie_id


class _FakeNode:
    def __init__(self, objs, idx=0):
        self._objs = objs
        self._idx = idx
        self.data = objs[idx]

    @property
    def next(self):
        if self._idx + 1 >= len(self._objs):
            return None
        return _FakeNode(self._objs, self._idx + 1)


class _FakeFrameMeta:
    def __init__(self, objs):
        self.obj_meta_list = _FakeNode(objs) if objs else None


class _CastShim:
    @staticmethod
    def cast(data):
        return data


def _patch_pyds(monkeypatch):
    monkeypatch.setattr(pp_mod.pyds, "NvDsObjectMeta", _CastShim, raising=False)


MOUTH_BBOX = (480, 300, 520, 360)
TROLLEY_BBOX = (400, 300, 1000, 700)


def _frame(mouth_conf, trolley_conf):
    return _FakeFrameMeta([
        _FakeObj(0, mouth_conf, MOUTH_BBOX, track_id=290, gie_id=1),
        _FakeObj(1, trolley_conf, TROLLEY_BBOX, track_id=280, gie_id=1),
    ])


def test_bridge_passes_propagated_frames_after_confident_hit(tmp_path, monkeypatch):
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=1)
    t = 1000.0
    m1, t1, _ = proc._extract_detections(_frame(0.9, 0.8), t)
    assert len(m1) == 1 and len(t1) == 1
    # Propagated frame 40ms later: conf=-0.1 on both — must still pass via bridge.
    m2, t2, _ = proc._extract_detections(_frame(-0.1, -0.1), t + 0.04)
    assert len(m2) == 1 and len(t2) == 1


def test_bridge_rejects_propagated_frames_with_no_confident_history(tmp_path, monkeypatch):
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=1)
    m, tr, _ = proc._extract_detections(_frame(-0.1, -0.1), 1000.0)
    assert m == [] and tr == []


def test_bridge_expires_after_window(tmp_path, monkeypatch):
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=1)
    t = 1000.0
    proc._extract_detections(_frame(0.9, 0.8), t)
    # Well past the bridge window (bridge = max(0.5, 4*2/25) = 0.5s)
    m, tr, _ = proc._extract_detections(_frame(-0.1, -0.1), t + 5.0)
    assert m == [] and tr == []


def test_bridge_disabled_at_interval_zero(tmp_path, monkeypatch):
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=0)
    t = 1000.0
    proc._extract_detections(_frame(0.9, 0.8), t)
    m, tr, _ = proc._extract_detections(_frame(-0.1, -0.1), t + 0.04)
    assert m == [] and tr == []  # original strict-gate semantics preserved


def test_session_starts_despite_alternating_propagated_frames(tmp_path, monkeypatch):
    """End-to-end regression: the 15:42 missed-pour scenario. Alternating
    confident/propagated frames at 25fps must accumulate to session start."""
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=1)
    proc._save_event_screenshot = lambda *a, **k: None
    from datetime import datetime
    t0 = 1000.0
    for i in range(40):  # 1.6s at 25fps
        conf = 0.9 if i % 2 == 0 else -0.1
        proc.process_frame(
            _frame(conf, 0.8 if i % 2 == 0 else -0.1),
            frame=None,
            timestamp=t0 + i * 0.04,
            datetime_obj=datetime.now(),
        )
    assert proc.session_active, "session must start within 1.6s of alternating frames"


def test_session_never_starts_without_fix_semantics_at_interval_zero(tmp_path, monkeypatch):
    """Control: at interval=0 (no bridge), truly absent mouths still block sessions."""
    _patch_pyds(monkeypatch)
    proc = _make_proc(tmp_path, interval=0)
    proc._save_event_screenshot = lambda *a, **k: None
    from datetime import datetime
    t0 = 1000.0
    for i in range(40):
        conf = 0.9 if i % 2 == 0 else -0.1  # alternating: strict gates drop half
        proc.process_frame(
            _frame(conf, 0.8),
            frame=None,
            timestamp=t0 + i * 0.04,
            datetime_obj=datetime.now(),
        )
    assert not proc.session_active
