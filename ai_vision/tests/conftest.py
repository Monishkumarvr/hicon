import sys
import types


def _install_fake_pyds():
    fake = types.ModuleType("pyds")

    class _ObjMeta:
        @staticmethod
        def cast(data):
            return data

    fake.NvDsObjectMeta = _ObjMeta
    fake.nvds_acquire_display_meta_from_pool = lambda batch_meta: None
    fake.nvds_add_display_meta_to_frame = lambda frame_meta, display_meta: None
    sys.modules["pyds"] = fake


try:
    import pyds  # noqa: F401
except Exception:
    _install_fake_pyds()
