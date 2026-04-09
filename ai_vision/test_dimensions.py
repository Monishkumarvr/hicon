import numpy as np

def _measure_multi_probe_brightness_nv12(frame, base_x, base_y):
    visible_h = frame.shape[0] * 2 // 3
    w = frame.shape[1]
    print(f"Frame shape from nvds_buf_surface (ndim=2): {frame.shape}, parsed w: {w}, parsed visible_h: {visible_h}")
    
# test
f = np.zeros((1350, 1600), dtype=np.uint8)
_measure_multi_probe_brightness_nv12(f, 0, 0)
