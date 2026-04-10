"""
DeepStream Pipeline Builder - HiCon 3-Stream Architecture
Constructs DS 7.1 pipeline for induction furnace monitoring:
  Stream 0 (Process Camera):   pouring detection (nvinfer GIE-1) + brightness analysis (probe)
  Stream 1 (Pyrometer Camera): rod detection (nvinfer GIE-2)
  Stream 2 (Pouring2 Camera):  pouring detection (nvinfer GIE-3, no brightness)
All cameras use H.265/HEVC; per-stream codec is configurable via 'rtsp_codec_N' config keys.
"""
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GstRtsp

logger = logging.getLogger(__name__)
_NVDSOSD_MODE_CPU = 0
_NVDSOSD_MODE_GPU = 1


class DeepStreamPipelineBuilder:
    """
    Builds DeepStream 7.1 3-stream pipeline for HiCon furnace monitoring.

    Architecture:
    - Stream 0 → nvv4l2decoder(H265) → mux_0 → nvinfer(pouring,GIE-1) → nvtracker → nvosd → sink_0
    - Stream 1 → nvv4l2decoder(H265) → mux_1 → nvinfer(pyrometer,GIE-2) → nvosd → sink_1
    - Stream 2 → nvv4l2decoder(H265) → mux_2 → nvinfer(pouring,GIE-3) → nvtracker → nvosd → sink_2
    """

    def __init__(self, config: dict):
        """
        Initialize pipeline builder.

        Args:
            config: Configuration dictionary with:
                - rtsp_stream_0: RTSP URL for Process camera
                - rtsp_stream_1: RTSP URL for Pyrometer camera
                - config_pouring: Path to pouring nvinfer config
                - config_pyrometer: Path to pyrometer nvinfer config
                - tracker_lib: Path to tracker library
                - tracker_config: Path to tracker config
                - stream_0_tracker_config: Optional Stream 0 tracker config override
        """
        self.config = config
        self.pipeline = None
        self.elements = {}
        self.enable_inference_video = bool(config.get('enable_inference_video', False))
        self.enable_inference_video_stream_0 = bool(
            config.get('enable_inference_video_stream_0', self.enable_inference_video)
        )
        self.enable_inference_video_stream_1 = bool(
            config.get('enable_inference_video_stream_1', self.enable_inference_video)
        )
        self.enable_inference_video_stream_2 = bool(
            config.get('enable_inference_video_stream_2', self.enable_inference_video)
        )
        self.enable_live_stream_0 = bool(config.get('enable_live_stream_0', False))
        self.enable_stream0_local_relay = bool(config.get('enable_stream0_local_relay', False))
        self.stream0_bypass_pgie = bool(config.get('stream_0_bypass_pgie', False))
        self.stream0_bypass_tracker = bool(config.get('stream_0_bypass_tracker', False))
        self.stream0_decode_only_mode = bool(config.get('stream_0_decode_only_mode', False))
        self.stream0_postmux_only_mode = bool(config.get('stream_0_postmux_only_mode', False))
        self.stream0_postconv_only_mode = bool(config.get('stream_0_postconv_only_mode', False))
        self.stream0_preosd_only_mode = bool(config.get('stream_0_preosd_only_mode', False))
        self.stream0_decoupled_analysis_mode = bool(
            config.get('stream_0_decoupled_analysis_mode', False)
        )
        self.stream0_analysis_branch_enabled = bool(
            config.get('stream_0_analysis_branch_enabled', True)
        )
        self.stream0_analysis_rgba_enabled = bool(
            config.get('stream_0_analysis_rgba_enabled', True)
        )
        self.stream0_analysis_probe_enabled = bool(
            config.get('stream_0_analysis_probe_enabled', True)
        )
        self.stream0_mux_width = int(config.get('stream_0_mux_width', 1280) or 1280)
        self.stream0_mux_height = int(config.get('stream_0_mux_height', 720) or 720)
        self.stream0_tracker_width = int(config.get('stream_0_tracker_width', 640) or 640)
        self.stream0_tracker_height = int(config.get('stream_0_tracker_height', 384) or 384)
        self.stream0_tracker_config = str(
            config.get('stream_0_tracker_config', config.get('tracker_config', '')) or ''
        )
        self.use_safe_cuda_brightness = bool(
            config.get('use_safe_cuda_brightness', False)
        )
        self.stream0_melting_config_ini = str(
            config.get('stream_0_melting_config_ini', '') or ''
        )
        self.use_segment_buffer_0 = bool(config.get('use_segment_buffer_0', False))
        self.segment_buffer_dir_0 = str(
            config.get('segment_buffer_dir_0', '/dev/shm/hicon/stream0-buffer')
        )
        self.segment_buffer_segment_sec_0 = int(config.get('segment_buffer_segment_sec_0', 2) or 2)
        self.segment_buffer_delay_sec_0 = int(config.get('segment_buffer_delay_sec_0', 60) or 60)
        self.segment_buffer_retention_sec_0 = int(
            config.get('segment_buffer_retention_sec_0', 120) or 120
        )
        self.use_segment_buffer_2 = bool(config.get('use_segment_buffer_2', False))
        self.segment_buffer_dir_2 = str(
            config.get('segment_buffer_dir_2', '/dev/shm/hicon/stream2-buffer')
        )
        self.segment_buffer_segment_sec_2 = int(config.get('segment_buffer_segment_sec_2', 2) or 2)
        self.segment_buffer_delay_sec_2 = int(config.get('segment_buffer_delay_sec_2', 120) or 120)
        self.segment_buffer_retention_sec_2 = int(
            config.get('segment_buffer_retention_sec_2', 180) or 180
        )
        self.use_ffmpeg_src_0 = bool(config.get('use_ffmpeg_src_0', False))
        self.use_ffmpeg_src_2 = bool(config.get('use_ffmpeg_src_2', False))
        self.use_udp_loopback_0 = bool(config.get('use_udp_loopback_0', False))
        self.use_udp_loopback_2 = bool(config.get('use_udp_loopback_2', False))
        self._ffmpeg_procs = {}  # {stream_id: subprocess.Popen}
        self._source_fds = []
        self.use_nvurisrcbin_0 = bool(config.get('use_nvurisrcbin_0', False))
        self.use_nvurisrcbin_1 = bool(config.get('use_nvurisrcbin_1', False))
        self.use_nvurisrcbin_2 = bool(config.get('use_nvurisrcbin_2', False))
        if self.use_segment_buffer_0:
            self.use_udp_loopback_0 = False
            self.use_ffmpeg_src_0 = False
            self.use_nvurisrcbin_0 = False
        if self.use_segment_buffer_2:
            self.use_udp_loopback_2 = False
            self.use_ffmpeg_src_2 = False
        if self.use_udp_loopback_0:
            self.use_ffmpeg_src_0 = False   # UDP loopback takes priority
            self.use_nvurisrcbin_0 = False
        if self.use_ffmpeg_src_0:
            self.use_nvurisrcbin_0 = False  # ffmpeg pipe takes priority over nvurisrcbin
        if self.use_nvurisrcbin_0 and not Gst.ElementFactory.find("nvurisrcbin"):
            logger.warning("nvurisrcbin not available — falling back to rtspsrc for Stream 0")
            self.use_nvurisrcbin_0 = False
        if self.stream0_bypass_pgie and not self.stream0_bypass_tracker:
            logger.info("Stream 0: pgie bypass requested; tracker bypass forced on as well")
            self.stream0_bypass_tracker = True
        self.stream0_annotated_tee_enabled = bool(
            self.enable_inference_video_stream_0
            or self.enable_live_stream_0
            or self.enable_stream0_local_relay
        )

    def _is_native_rtsp_stream(self, stream_id):
        """True when the stream uses rtspsrc or nvurisrcbin directly."""
        if self._is_ffmpeg_stream(stream_id) or self._is_segment_buffer_stream(stream_id):
            return False
        if stream_id == 0:
            return not self.use_udp_loopback_0 and bool(self.config.get('rtsp_stream_0'))
        if stream_id == 1:
            return bool(self.config.get('rtsp_stream_1'))
        if stream_id == 2:
            return not self.use_udp_loopback_2 and bool(self.config.get('rtsp_stream_2'))
        return False

    def get_restartable_stream_ids(self):
        return {
            sid for sid in (0, 1, 2)
            if self._is_native_rtsp_stream(sid)
        }

    def schedule_stream_restart(self, stream_id, reason=""):
        """Cycle a native RTSP source through NULL -> READY -> PLAYING."""
        if not self._is_native_rtsp_stream(stream_id):
            logger.warning("Stream %s: local restart not supported (%s)", stream_id, reason)
            return False

        source = self.elements.get(f"source{stream_id}")
        if source is None:
            logger.error("Stream %s: source element missing; cannot restart (%s)", stream_id, reason)
            return False

        logger.warning("Stream %s: restarting native RTSP source (%s)", stream_id, reason)
        try:
            for state in (Gst.State.NULL, Gst.State.READY, Gst.State.PLAYING):
                source.set_state(state)
                try:
                    source.get_state(2 * Gst.SECOND)
                except Exception:
                    pass
            return True
        except Exception as exc:
            logger.error("Stream %s: local restart failed (%s): %s", stream_id, reason, exc, exc_info=True)
            return False

    def terminate_ffmpeg_procs(self):
        """Terminate all wrapper/helper subprocesses. Call on pipeline shutdown.

        Sends SIGTERM to the bash wrapper process group, which kills both
        the wrapper and the ffmpeg child. Falls back to SIGKILL after 5s.
        """
        import signal
        for sid, proc in self._ffmpeg_procs.items():
            if proc.poll() is None:
                logger.info(f"Stream {sid}: terminating ffmpeg wrapper (pid={proc.pid})")
                try:
                    # Kill entire process group (bash + ffmpeg child)
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Stream {sid}: ffmpeg wrapper did not exit, killing (pid={proc.pid})")
                    proc.kill()
        for fd in self._source_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._source_fds.clear()

    @staticmethod
    def _configure_queue(queue, max_buffers=16, leaky=0, max_size_time_ns=0):
        """Configure a small queue with bounded buffers and optional leak behavior."""
        if not queue:
            return
        queue.set_property('max-size-buffers', max_buffers)
        queue.set_property('max-size-bytes', 0)
        queue.set_property('max-size-time', max_size_time_ns)
        queue.set_property('leaky', leaky)

    @classmethod
    def _configure_leaky_queue(cls, queue, max_buffers=16, max_size_time_ns=0):
        """Configure a downstream-leaky queue to isolate transient backpressure."""
        cls._configure_queue(queue, max_buffers=max_buffers, leaky=2,
                             max_size_time_ns=max_size_time_ns)

    def create_pipeline(self):
        """
        Create complete 3-stream pipeline.

        Returns:
            Tuple of (pipeline, elements_dict) or (None, None) on failure
        """
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            logger.error("Unable to create Pipeline")
            return None, None

        if not self._create_all_elements():
            return None, None

        if not self._link_all_branches():
            return None, None

        logger.info("3-stream HiCon pipeline created successfully")
        return self.pipeline, self.elements

    def _create_streammux(self, name, batch_size=1, width=1280, height=720):
        """Create nvstreammux with standard properties."""
        mux = Gst.ElementFactory.make("nvstreammux", name)
        if not mux:
            logger.error(f"Failed to create nvstreammux: {name}")
            return None
        mux.set_property('batch-size', batch_size)
        mux.set_property('width', width)
        mux.set_property('height', height)
        mux.set_property('batched-push-timeout', 40000)
        mux.set_property('live-source', True)
        mux.set_property('sync-inputs', 0)
        mux.set_property('buffer-pool-size', 8)
        mux.set_property('enable-padding', True)
        return mux

    @staticmethod
    def _tune_stream0_mux_for_cp_plus(mux):
        """
        Stream 0 (CP Plus) stalls with nvstreammux defaults after a few minutes even in a
        standalone decode->mux->fakesink pipeline. Synchronous mux processing and a larger
        output pool were stable in direct gst-launch soak tests.
        """
        if not mux:
            return
        mux.set_property('async-process', False)
        mux.set_property('buffer-pool-size', 32)
        mux.set_property('attach-sys-ts', True)
        logger.info(
            "Stream 0 (CP Plus): tuned nvstreammux "
            "(async-process=false, buffer-pool-size=32, attach-sys-ts=true)"
        )

    @staticmethod
    def _tune_stream0_postmux_convert_for_cp_plus(convert):
        """
        Stream 0 (CP Plus) also stalls when the post-mux nvvideoconvert is inserted.
        Force the conversion off the Jetson default VIC path, disable passthrough,
        and give the element a deeper output pool.
        """
        if not convert:
            return

        for name, value in (
            ('compute-hw', 1),          # GPU
            ('copy-hw', 1),             # GPU
            ('output-buffers', 32),
            ('disable-passthrough', True),
        ):
            try:
                convert.set_property(name, value)
            except Exception as exc:
                logger.warning(f"Stream 0: failed to set nvvideoconvert {name}={value}: {exc}")

        logger.info(
            "Stream 0 (CP Plus): tuned post-mux nvvideoconvert "
            "(compute-hw=GPU, copy-hw=GPU, output-buffers=32, disable-passthrough=true)"
        )

    def _create_nvurisrcbin_chain(self, stream_id, rtsp_url):
        """Create nvurisrcbin source with built-in RTSP reconnection (replaces rtspsrc chain).

        nvurisrcbin encapsulates rtspsrc + depay + parser + decoder internally.
        Output: video/x-raw(memory:NVMM) on dynamic pad 'vsrc_0'.
        """
        sid = str(stream_id)
        uribin = Gst.ElementFactory.make("nvurisrcbin", f"source{sid}")
        if not uribin:
            logger.error(f"Stream {sid}: Failed to create nvurisrcbin")
            return False

        uribin.set_property('uri', rtsp_url)
        uribin.set_property('type', 2)  # rtsp source type
        # Stream 0 (1280x720) needs longer reconnect interval: decoder DPB reallocation
        # + TCP socket cleanup takes >2s at higher resolution.  640x480 streams recover
        # fast enough with 2s.  10s matches the nvurisrcbin default and avoids the
        # "reconnect loop" where each attempt interrupts the previous one.
        reconnect_s = 10 if stream_id == 0 else 2
        uribin.set_property('rtsp-reconnect-interval', reconnect_s)
        uribin.set_property('rtsp-reconnect-attempts', -1)  # unlimited
        uribin.set_property('gpu-id', 0)
        uribin.set_property('cudadec-memtype', 0)  # device memory (NVMM)
        uribin.set_property('num-extra-surfaces', 16)

        latency_ms = int(self.config.get('rtsp_latency_ms', 4000) or 4000)
        uribin.set_property('latency', latency_ms)
        uribin.set_property('drop-on-latency', True)

        # For nvurisrcbin, prefer auto (UDP fallback) — TCP-only reconnection is slower
        # because of TCP socket TIME-WAIT cleanup.  Override any per-stream tcp setting.
        protocol = str(self.config.get(f'rtsp_protocol_{stream_id}', 'auto') or 'auto').lower()
        if protocol == 'tcp' and stream_id == 0:
            logger.info(f"Stream {stream_id}: overriding protocol tcp→auto for nvurisrcbin (faster reconnect)")
            protocol = 'auto'
        if protocol == 'tcp':
            uribin.set_property('select-rtp-protocol', 4)  # rtp-tcp
        # default 0 = rtp-multi (UDP + UDP Multicast + TCP)

        self.elements[f'source{sid}'] = uribin

        # Leaky queue for ALL streams to absorb backpressure from downstream
        # inference spikes — prevents RTSP TCP socket stall → server disconnect.
        # Stream 0 had this since March 6 investigation; Streams 1 & 2 were missing it.
        queue_name = f'premuxq{stream_id}'
        self.elements[queue_name] = Gst.ElementFactory.make("queue", queue_name)
        self._configure_leaky_queue(self.elements[queue_name], max_buffers=128,
                                    max_size_time_ns=5_000_000_000)

        logger.info(
            f"Stream {sid}: nvurisrcbin created "
            f"(reconnect-interval={reconnect_s}s, reconnect-attempts=unlimited, "
            f"protocol={'tcp' if protocol == 'tcp' else 'multi'}, latency={latency_ms}ms, "
            f"drop-on-latency=True, num-extra-surfaces=16, "
            f"premuxq={queue_name} leaky=2 max-buffers=128 max-time=5s)"
        )
        return True

    def _create_ffmpeg_chain(self, stream_id, rtsp_url):
        """Create ffmpeg subprocess → fdsrc → parse → caps → decode → nvvidconv → caps chain.

        ffmpeg handles RTSP transport and keepalives (zero drops on CP Plus NVR).
        It remuxes only (-c:v copy, ~1% CPU) and pipes raw H.264/H.265 byte stream
        to GStreamer's fdsrc via stdout. NVDEC does the single HW decode.
        """
        sid = str(stream_id)
        codec = str(self.config.get(f'rtsp_codec_{stream_id}', 'h265')).lower()
        container_fmt = 'hevc' if codec == 'h265' else 'h264'

        # Bash wrapper auto-restarts ffmpeg on connection loss.
        # The bash process holds the pipe open — fdsrc never sees EOF.
        # When ffmpeg exits (NVR drops session), bash logs, sleeps 2s, relaunches.
        # Recovery: ~5s (2s sleep + ~3s connect) vs nvurisrcbin's ~35s blind time.
        rtsp_proto = str(self.config.get(f'rtsp_protocol_{stream_id}', 'udp')).lower()
        # UDP: large socket buffer + reorder tolerance to absorb jitter.
        # Pipe through dd (4MB buffer, iflag=fullblock) to decouple ffmpeg's
        # network read loop from downstream pipe backpressure — prevents UDP
        # packet loss when GStreamer stalls briefly on inference spikes.
        if rtsp_proto == 'udp':
            transport_opts = (
                f'-rtsp_transport udp -buffer_size 4194304 '
                f'-max_delay 500000 -reorder_queue_size 2000'
            )
        else:
            transport_opts = f'-rtsp_transport {rtsp_proto}'
        ffmpeg_cmd = (
            f'ffmpeg -hide_banner -loglevel warning '
            f'{transport_opts} -stimeout 10000000 '
            f"-i '{rtsp_url}' -c:v copy -an -f {container_fmt} pipe:1"
        )
        wrapper_script = (
            f'while true; do '
            f'echo "Stream {sid}: ffmpeg starting ({rtsp_proto})" >&2; '
            f'{ffmpeg_cmd}; '
            f'EXIT_CODE=$?; '
            f'echo "Stream {sid}: ffmpeg exited (code=$EXIT_CODE), restarting in 2s..." >&2; '
            f'sleep 2; '
            f'done'
        )
        try:
            proc = subprocess.Popen(
                ['bash', '-c', wrapper_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error(f"Stream {sid}: bash/ffmpeg not found — cannot use ffmpeg source")
            return False
        except Exception as exc:
            logger.error(f"Stream {sid}: failed to launch ffmpeg wrapper: {exc}")
            return False

        self._ffmpeg_procs[stream_id] = proc

        # Enlarge the pipe buffer to 1MB to prevent back-pressure from blocking
        # ffmpeg's write loop (which would stall its RTSP read → NVR drops session).
        try:
            import fcntl
            _F_SETPIPE_SZ = 1031
            fcntl.fcntl(proc.stdout.fileno(), _F_SETPIPE_SZ, 1048576)
            logger.info(f"Stream {sid}: pipe buffer enlarged to 1MB")
        except Exception as exc:
            logger.debug(f"Stream {sid}: could not enlarge pipe buffer: {exc}")

        # Daemon thread to drain ffmpeg stderr and log warnings
        def _drain_stderr(p, s_id):
            try:
                for line in p.stderr:
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        logger.warning(f"Stream {s_id} ffmpeg: {text}")
            except Exception:
                pass
        t = threading.Thread(target=_drain_stderr, args=(proc, sid), daemon=True)
        t.start()

        # fdsrc reads from ffmpeg stdout pipe
        fdsrc = Gst.ElementFactory.make("fdsrc", f"source{sid}")
        if not fdsrc:
            logger.error(f"Stream {sid}: failed to create fdsrc")
            proc.kill()
            return False
        fdsrc.set_property('fd', proc.stdout.fileno())
        fdsrc.set_property('do-timestamp', True)
        fdsrc.set_property('blocksize', 65536)
        self.elements[f'source{sid}'] = fdsrc

        # Decoupling queue after fdsrc prevents downstream stalls from
        # back-pressuring the pipe (which would block ffmpeg and stall RTSP).
        self.elements[f'ffmpegq{sid}'] = Gst.ElementFactory.make("queue", f"ffmpegq{sid}")
        self._configure_leaky_queue(self.elements[f'ffmpegq{sid}'], max_buffers=64)

        # Parser + caps filter (same as rtspsrc decode chain)
        if codec == 'h265':
            parser = Gst.ElementFactory.make("h265parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h265, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter
        else:
            parser = Gst.ElementFactory.make("h264parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
                parser.set_property('disable-passthrough', True)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h264, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter

        # Decoder + convert + NVMM caps (same as rtspsrc path)
        self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("nvv4l2decoder", f"decoder{sid}")
        if self.elements[f'decoder{sid}']:
            decoder = self.elements[f'decoder{sid}']
            decoder.set_property('drop-frame-interval', 0)
            decoder.set_property('num-extra-surfaces', 8)
            if stream_id == 0:
                try:
                    decoder.set_property('disable-dpb', True)
                except Exception:
                    pass

        self.elements[f'nvvidconv{sid}'] = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv{sid}")
        self.elements[f'caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"caps{sid}")
        if self.elements[f'caps{sid}']:
            self.elements[f'caps{sid}'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
            )

        # premuxq0 for Stream 0 isolation (same as other source paths)
        if stream_id == 0:
            self.elements['premuxq0'] = Gst.ElementFactory.make("queue", "premuxq0")
            self._configure_leaky_queue(self.elements['premuxq0'])

        logger.info(
            f"Stream {sid}: ffmpeg auto-restart wrapper created "
            f"(codec={codec}, wrapper_pid={proc.pid}, fd={proc.stdout.fileno()})"
        )
        return True

    def _create_udp_loopback_chain(self, stream_id, rtsp_url, udp_port):
        """Create UDP loopback bridge: ffmpeg → MPEGTS → UDP localhost → udpsrc → tsdemux → parse → decode.

        Unlike the fdsrc pipe bridge, ffmpeg's UDP sendto() never blocks regardless of downstream
        backpressure. This prevents the pipeline's inference load from stalling ffmpeg's TCP read
        loop, which would trigger the CP Plus camera's ~5-min RTSP session timeout.

        Architecture:
            camera (TCP) → ffmpeg (-c:v copy → MPEGTS) → UDP 127.0.0.1:{port}
                                                               ↓
            udpsrc → tsdemux → h264parse → nvv4l2decoder → nvvideoconvert → NVMM NV12
        """
        sid = str(stream_id)
        rtsp_proto = str(self.config.get(f'rtsp_protocol_{stream_id}', 'tcp')).lower()
        codec = str(self.config.get(f'rtsp_codec_{stream_id}', 'h265')).lower()

        ffmpeg_cmd = (
            f'nice -n -10 ffmpeg -hide_banner -loglevel warning '
            f'-rtsp_transport {rtsp_proto} -stimeout 10000000 '
            f"-i '{rtsp_url}' -c:v copy -an -f mpegts "
            f"'udp://127.0.0.1:{udp_port}?pkt_size=1316'"
        )
        wrapper_script = (
            f'while true; do '
            f'echo "Stream {sid}: ffmpeg UDP loopback starting (port={udp_port})" >&2; '
            f'{ffmpeg_cmd}; '
            f'EXIT_CODE=$?; '
            f'echo "Stream {sid}: ffmpeg exited (code=$EXIT_CODE), restarting in 2s..." >&2; '
            f'sleep 2; '
            f'done'
        )
        try:
            proc = subprocess.Popen(
                ['bash', '-c', wrapper_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error(f"Stream {sid}: bash/ffmpeg not found — cannot use UDP loopback source")
            return False
        except Exception as exc:
            logger.error(f"Stream {sid}: failed to launch UDP loopback ffmpeg: {exc}")
            return False

        self._ffmpeg_procs[stream_id] = proc

        def _drain_stderr(p, s_id):
            try:
                for line in p.stderr:
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        logger.warning(f"Stream {s_id} ffmpeg: {text}")
            except Exception:
                pass
        t = threading.Thread(target=_drain_stderr, args=(proc, sid), daemon=True)
        t.start()

        # udpsrc reads MPEGTS from localhost UDP port
        udpsrc = Gst.ElementFactory.make("udpsrc", f"source{sid}")
        if not udpsrc:
            logger.error(f"Stream {sid}: failed to create udpsrc")
            proc.kill()
            return False
        udpsrc.set_property('port', udp_port)
        udpsrc.set_property('buffer-size', 8388608)  # 8MB kernel UDP recv buffer
        udpsrc.set_property('caps', Gst.Caps.from_string("video/mpegts"))
        self.elements[f'source{sid}'] = udpsrc

        # tsdemux extracts the H.264/H.265 elementary stream from MPEGTS
        tsdemux = Gst.ElementFactory.make("tsdemux", f"tsdemux{sid}")
        if not tsdemux:
            logger.error(f"Stream {sid}: failed to create tsdemux")
            proc.kill()
            return False
        self.elements[f'tsdemux{sid}'] = tsdemux

        # Queue between tsdemux (dynamic pad) and parser
        self.elements[f'demuxq{sid}'] = Gst.ElementFactory.make("queue", f"demuxq{sid}")
        self._configure_leaky_queue(self.elements[f'demuxq{sid}'], max_buffers=64)

        # Parser + caps (same as fdsrc path)
        if codec == 'h265':
            parser = Gst.ElementFactory.make("h265parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h265, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter
        else:
            parser = Gst.ElementFactory.make("h264parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
                parser.set_property('disable-passthrough', True)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h264, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter

        # Decoder + convert + NVMM caps
        self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("nvv4l2decoder", f"decoder{sid}")
        if self.elements[f'decoder{sid}']:
            decoder = self.elements[f'decoder{sid}']
            decoder.set_property('drop-frame-interval', 0)
            decoder.set_property('num-extra-surfaces', 8)
            if stream_id == 0:
                try:
                    decoder.set_property('disable-dpb', True)
                except Exception:
                    pass

        self.elements[f'nvvidconv{sid}'] = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv{sid}")
        self.elements[f'caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"caps{sid}")
        if self.elements[f'caps{sid}']:
            self.elements[f'caps{sid}'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
            )

        if stream_id == 0:
            self.elements['premuxq0'] = Gst.ElementFactory.make("queue", "premuxq0")
            self._configure_leaky_queue(self.elements['premuxq0'])

        logger.info(
            f"Stream {sid}: UDP loopback source created "
            f"(codec={codec}, port={udp_port}, ffmpeg_pid={proc.pid})"
        )
        return True

    def _create_segment_buffer_chain(
        self,
        stream_id,
        rtsp_url,
        buffer_dir,
        segment_sec,
        delay_sec,
        retention_sec,
    ):
        """Create delayed segment-buffer ingest: helper → FIFO → h264parse → decode."""
        sid = str(stream_id)
        codec = str(self.config.get(f'rtsp_codec_{stream_id}', 'h265')).lower()
        fifo_path = Path(buffer_dir) / "stream.fifo"
        helper_path = Path(__file__).with_name("segment_buffer_helper.py")

        # Delete any FIFO left from a previous run BEFORE spawning the helper.
        # If we open the old FIFO first, helper's shutil.rmtree deletes it, leaving
        # fdsrc reading from a deleted inode with no writer — blocks forever.
        if fifo_path.exists():
            try:
                fifo_path.unlink()
            except OSError:
                pass
        fps = float(self.config.get(f'rtsp_fps_{stream_id}', 25.0) or 25.0)
        helper_cmd = [
            sys.executable,
            str(helper_path),
            "--stream-id",
            sid,
            "--rtsp-url",
            rtsp_url,
            "--codec",
            codec,
            "--fps",
            str(fps),
            "--buffer-dir",
            str(buffer_dir),
            "--segment-seconds",
            str(segment_sec),
            "--delay-seconds",
            str(delay_sec),
            "--retention-seconds",
            str(retention_sec),
        ]
        try:
            proc = subprocess.Popen(
                helper_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                start_new_session=True,
            )
        except FileNotFoundError:
            logger.error(f"Stream {sid}: python3 not found — cannot use segment buffer source")
            return False
        except Exception as exc:
            logger.error(f"Stream {sid}: failed to launch segment buffer helper: {exc}")
            return False

        self._ffmpeg_procs[stream_id] = proc

        def _drain_stderr(p, s_id):
            try:
                for line in p.stderr:
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        logger.info(f"Stream {s_id} segment-buffer: {text}")
            except Exception:
                pass

        t = threading.Thread(target=_drain_stderr, args=(proc, sid), daemon=True)
        t.start()

        deadline = time.time() + 5.0
        while not fifo_path.exists():
            if proc.poll() is not None:
                logger.error(f"Stream {sid}: segment buffer helper exited before FIFO was created")
                return False
            if time.time() >= deadline:
                logger.error(f"Stream {sid}: timed out waiting for segment buffer FIFO at {fifo_path}")
                return False
            time.sleep(0.05)

        try:
            import fcntl

            fifo_fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            flags = fcntl.fcntl(fifo_fd, fcntl.F_GETFL)
            fcntl.fcntl(fifo_fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        except OSError as exc:
            logger.error(f"Stream {sid}: failed to open segment buffer FIFO {fifo_path}: {exc}")
            return False
        self._source_fds.append(fifo_fd)

        fdsrc = Gst.ElementFactory.make("fdsrc", f"source{sid}")
        if not fdsrc:
            logger.error(f"Stream {sid}: failed to create fdsrc")
            return False
        fdsrc.set_property('fd', fifo_fd)
        fdsrc.set_property('do-timestamp', True)
        fdsrc.set_property('blocksize', 65536)
        self.elements[f'source{sid}'] = fdsrc

        # Use a leaky decoupling queue after fdsrc (same as ffmpeg pipe path).
        # Raw H264 segments from the helper feed directly into h264parse — no MPEGTS
        # demuxing needed, so no tsdemux dynamic-pad deadlock.
        self.elements[f'segbufq{sid}'] = Gst.ElementFactory.make("queue", f"segbufq{sid}")
        self._configure_leaky_queue(self.elements[f'segbufq{sid}'], max_buffers=64)

        if codec == 'h265':
            parser = Gst.ElementFactory.make("h265parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h265, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter
        else:
            parser = Gst.ElementFactory.make("h264parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
                parser.set_property('disable-passthrough', True)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h264, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter

        self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("nvv4l2decoder", f"decoder{sid}")
        if self.elements[f'decoder{sid}']:
            decoder = self.elements[f'decoder{sid}']
            decoder.set_property('drop-frame-interval', 0)
            decoder.set_property('num-extra-surfaces', 8)
            if stream_id == 0:
                try:
                    decoder.set_property('disable-dpb', True)
                except Exception:
                    pass

        self.elements[f'nvvidconv{sid}'] = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv{sid}")
        self.elements[f'caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"caps{sid}")
        if self.elements[f'caps{sid}']:
            self.elements[f'caps{sid}'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
            )

        if stream_id == 0:
            self.elements['premuxq0'] = Gst.ElementFactory.make("queue", "premuxq0")
            self._configure_queue(self.elements['premuxq0'], max_buffers=64, leaky=0)

        logger.info(
            f"Stream {sid}: segment buffer source created "
            f"(codec={codec}, dir={buffer_dir}, delay={delay_sec}s, retention={retention_sec}s)"
        )
        return True

    def _create_decode_chain(self, stream_id, rtsp_url):
        """Create RTSP source → depay → parse → caps → decode → nvvidconv → caps chain.

        Codec is determined by config key 'rtsp_codec_{stream_id}' ('h265' or 'h264').
        Element keys use generic names (parser{sid}, vidcaps{sid}) so _link_decode_chain
        works uniformly for both codecs without conditional branching.
        nvv4l2decoder auto-detects codec from upstream caps — no extra configuration needed.
        """
        sid = str(stream_id)
        codec = str(self.config.get(f'rtsp_codec_{stream_id}', 'h265')).lower()

        self.elements[f'source{sid}'] = Gst.ElementFactory.make("rtspsrc", f"source{sid}")
        self._configure_rtsp_source(self.elements[f'source{sid}'], rtsp_url, stream_id)
        if stream_id == 0:
            self.elements['srcq0'] = Gst.ElementFactory.make("queue", "srcq0")
            self._configure_leaky_queue(self.elements['srcq0'])

        if codec == 'h265':
            self.elements[f'depay{sid}'] = Gst.ElementFactory.make("rtph265depay", f"depay{sid}")
            parser = Gst.ElementFactory.make("h265parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h265, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter
        else:
            # h264 fallback
            self.elements[f'depay{sid}'] = Gst.ElementFactory.make("rtph264depay", f"depay{sid}")
            parser = Gst.ElementFactory.make("h264parse", f"parser{sid}")
            if parser:
                parser.set_property('config-interval', -1)
                parser.set_property('disable-passthrough', True)
            self.elements[f'parser{sid}'] = parser
            caps_filter = Gst.ElementFactory.make("capsfilter", f"vidcaps{sid}")
            if caps_filter:
                caps_filter.set_property(
                    "caps",
                    Gst.Caps.from_string("video/x-h264, stream-format=byte-stream, alignment=au")
                )
            self.elements[f'vidcaps{sid}'] = caps_filter

        self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("nvv4l2decoder", f"decoder{sid}")
        if self.elements[f'decoder{sid}']:
            decoder = self.elements[f'decoder{sid}']
            decoder.set_property('num-extra-surfaces', 8)
            decoder.set_property('enable-max-performance', True)
            decoder.set_property('enable-error-check', True)
            # enable-frame-type-reporting removed: diagnostic confirmed no B-frames
            if stream_id == 0:
                decoder.set_property('disable-dpb', True)
                logger.info(
                    f"Stream {sid}: Using nvv4l2decoder (codec={codec.upper()}, "
                    f"disable-dpb=true, enable-max-performance=true, enable-error-check=true)"
                )
            else:
                logger.info(f"Stream {sid}: Using nvv4l2decoder (codec={codec.upper()})")
        else:
            logger.error(f"Stream {sid}: nvv4l2decoder unavailable — no reliable SW H.265 fallback")

        self.elements[f'nvvidconv{sid}'] = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv{sid}")
        self.elements[f'caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"caps{sid}")
        if self.elements[f'caps{sid}']:
            self.elements[f'caps{sid}'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
            )
        if stream_id == 0:
            self.elements['premuxq0'] = Gst.ElementFactory.make("queue", "premuxq0")
            self._configure_leaky_queue(self.elements['premuxq0'])
            logger.info(
                "Stream 0 (CP Plus): source isolation queues enabled "
                "(srcq0, premuxq0, leaky=2, max-size-buffers=16)"
            )

        logger.info(f"Stream {sid}: Decode chain created (codec={codec.upper()})")

    def _create_all_elements(self):
        """Create all pipeline elements for 3-stream HiCon."""

        # Determine enabled streams
        self.enabled_streams = []
        for i in range(3):
            url = self.config.get(f'rtsp_stream_{i}', '')
            if url:
                self.enabled_streams.append(i)
            else:
                logger.info(f"Stream {i}: disabled (no URL)")

        if not self.enabled_streams:
            logger.error("No streams enabled - cannot create pipeline")
            return False

        # === STREAM 0: Process Camera ===
        if 0 in self.enabled_streams:
            if self.use_segment_buffer_0:
                if not self._create_segment_buffer_chain(
                    0,
                    self.config['rtsp_stream_0'],
                    self.segment_buffer_dir_0,
                    self.segment_buffer_segment_sec_0,
                    self.segment_buffer_delay_sec_0,
                    self.segment_buffer_retention_sec_0,
                ):
                    return False
            elif self.use_udp_loopback_0:
                port = self.config.get('udp_loopback_port_0', 5000)
                if not self._create_udp_loopback_chain(0, self.config['rtsp_stream_0'], port):
                    return False
            elif self.use_ffmpeg_src_0:
                if not self._create_ffmpeg_chain(0, self.config['rtsp_stream_0']):
                    return False
            elif self.use_nvurisrcbin_0:
                if not self._create_nvurisrcbin_chain(0, self.config['rtsp_stream_0']):
                    return False
            else:
                self._create_decode_chain(0, self.config['rtsp_stream_0'])

            if self.stream0_decode_only_mode:
                self.elements['decode_sink_0'] = Gst.ElementFactory.make("fakesink", "decode-sink-0")
                self.elements['decode_sink_0'].set_property('sync', 0)
                self.elements['decode_sink_0'].set_property('async', False)
                logger.warning(
                    "Stream 0 (CP Plus): decode-only diagnostic mode enabled "
                    "(bypassing mux, OSD, and recording path)"
                )
            elif self.stream0_postconv_only_mode:
                self.elements['mux_0'] = self._create_streammux(
                    "mux-0",
                    width=self.stream0_mux_width,
                    height=self.stream0_mux_height,
                )
                if not self.use_nvurisrcbin_0:
                    self._tune_stream0_mux_for_cp_plus(self.elements['mux_0'])
                else:
                    logger.info("Stream 0: skipping CP Plus mux tuning (async-process=False incompatible with nvurisrcbin reconnection)")
                self.elements['postmuxq0'] = Gst.ElementFactory.make("queue", "postmuxq0")
                self._configure_leaky_queue(self.elements['postmuxq0'], max_buffers=64)
                self.elements['nvvidconv_osd_0'] = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-osd-0")
                self._tune_stream0_postmux_convert_for_cp_plus(self.elements['nvvidconv_osd_0'])
                self.elements['postconv_sink_0'] = Gst.ElementFactory.make("fakesink", "postconv-sink-0")
                self.elements['postconv_sink_0'].set_property('sync', 0)
                self.elements['postconv_sink_0'].set_property('async', False)
                logger.info(
                    "Stream 0 (CP Plus): post-convert isolation queue enabled "
                    "(postmuxq0, leaky=2, max-size-buffers=16)"
                )
                logger.warning(
                    "Stream 0 (CP Plus): post-convert-only diagnostic mode enabled "
                    "(isolating nvvideoconvert from RGBA caps, nvdsosd, and downstream path)"
                )
            elif self.stream0_preosd_only_mode:
                self.elements['mux_0'] = self._create_streammux(
                    "mux-0",
                    width=self.stream0_mux_width,
                    height=self.stream0_mux_height,
                )
                if not self.use_nvurisrcbin_0:
                    self._tune_stream0_mux_for_cp_plus(self.elements['mux_0'])
                self.elements['postmuxq0'] = Gst.ElementFactory.make("queue", "postmuxq0")
                self._configure_leaky_queue(self.elements['postmuxq0'], max_buffers=64)
                self.elements['nvvidconv_osd_0'] = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-osd-0")
                self._tune_stream0_postmux_convert_for_cp_plus(self.elements['nvvidconv_osd_0'])
                self.elements['caps_osd_0'] = Gst.ElementFactory.make("capsfilter", "caps-osd-0")
                self.elements['caps_osd_0'].set_property(
                    'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                )
                self.elements['preosdq0'] = Gst.ElementFactory.make("queue", "preosdq0")
                self._configure_leaky_queue(self.elements['preosdq0'])
                self.elements['preosd_sink_0'] = Gst.ElementFactory.make("fakesink", "preosd-sink-0")
                self.elements['preosd_sink_0'].set_property('sync', 0)
                self.elements['preosd_sink_0'].set_property('async', False)
                logger.info(
                    "Stream 0 (CP Plus): pre-OSD isolation queues enabled "
                    "(postmuxq0, preosdq0, leaky=2, max-size-buffers=16)"
                )
                logger.warning(
                    "Stream 0 (CP Plus): pre-OSD-only diagnostic mode enabled "
                    "(isolating RGBA conversion from nvdsosd and downstream path)"
                )
            elif self.stream0_postmux_only_mode:
                self.elements['mux_0'] = self._create_streammux(
                    "mux-0",
                    width=self.stream0_mux_width,
                    height=self.stream0_mux_height,
                )
                if not self.use_nvurisrcbin_0:
                    self._tune_stream0_mux_for_cp_plus(self.elements['mux_0'])
                self.elements['postmuxq0'] = Gst.ElementFactory.make("queue", "postmuxq0")
                self._configure_leaky_queue(self.elements['postmuxq0'], max_buffers=64)
                self.elements['postmux_sink_0'] = Gst.ElementFactory.make("fakesink", "postmux-sink-0")
                self.elements['postmux_sink_0'].set_property('sync', 0)
                self.elements['postmux_sink_0'].set_property('async', False)
                logger.warning(
                    "Stream 0 (CP Plus): post-mux-only diagnostic mode enabled "
                    "(isolating nvstreammux from OSD/render and recording path)"
                )
            else:
                self.elements['mux_0'] = self._create_streammux(
                    "mux-0",
                    width=self.stream0_mux_width,
                    height=self.stream0_mux_height,
                )
                if not self.use_nvurisrcbin_0:
                    self._tune_stream0_mux_for_cp_plus(self.elements['mux_0'])
                else:
                    logger.info("Stream 0: skipping CP Plus mux tuning for nvurisrcbin (async-process=False blocks reconnection)")
                self.elements['postmuxq0'] = Gst.ElementFactory.make("queue", "postmuxq0")
                self._configure_leaky_queue(self.elements['postmuxq0'], max_buffers=64)

                # Pouring inference (GIE-1)
                if not self.stream0_bypass_pgie:
                    self.elements['pgie_pouring'] = Gst.ElementFactory.make("nvinfer", "pgie-pouring")
                    self.elements['pgie_pouring'].set_property('config-file-path', self.config['config_pouring'])
                    logger.info("Stream 0: Pouring nvinfer created (GIE-1)")

                # Tracker for pouring
                if not self.stream0_bypass_tracker:
                    self.elements['tracker_0'] = Gst.ElementFactory.make("nvtracker", "tracker-0")
                    self.elements['tracker_0'].set_property('ll-lib-file', self.config['tracker_lib'])
                    self.elements['tracker_0'].set_property('ll-config-file', self.stream0_tracker_config)
                    self.elements['tracker_0'].set_property('tracker-width', self.stream0_tracker_width)
                    self.elements['tracker_0'].set_property('tracker-height', self.stream0_tracker_height)

                self.elements['nvosd_0'] = Gst.ElementFactory.make("nvdsosd", "nvosd-0")
                if self.elements['nvosd_0']:
                    # CPU mode is more stable on the headless inference/recording path when
                    # Python processors attach display_meta with lines/rects/text.
                    self.elements['nvosd_0'].set_property('process-mode', _NVDSOSD_MODE_CPU)
                    if self.stream0_decoupled_analysis_mode:
                        # Stream 0 decoupled mode keeps the display path NV12-only for stability.
                        # On this path, attempting to draw PGIE/tracker rectangles has been the
                        # live failure boundary ("Unable to draw rectangles"). Keep the element in
                        # place for downstream topology, but turn off bbox/text rendering.
                        self.elements['nvosd_0'].set_property('display-bbox', False)
                        self.elements['nvosd_0'].set_property('display-text', False)
                if self.stream0_decoupled_analysis_mode:
                    # In decoupled mode, no nvvideoconvert is placed on Stream 0.
                    # Any nvvideoconvert on Stream 0 combined with a pre-OSD tee causes
                    # pgie-pyrometer CUDA OOM during TRT startup — keep Stream 0 NV12-only.
                    # nvosd_0 CPU mode handles NV12 directly; brightness probe uses NV12 Y-plane.
                    self.elements['tee_stream0_analysis'] = Gst.ElementFactory.make(
                        "tee", "tee-stream0-analysis"
                    )
                    self.elements['displayq0'] = Gst.ElementFactory.make("queue", "displayq0")
                    self._configure_queue(self.elements['displayq0'], max_buffers=16, leaky=0)
                    if self.stream0_analysis_branch_enabled:
                        self.elements['analysisq0'] = Gst.ElementFactory.make("queue", "analysisq0")
                        self._configure_queue(self.elements['analysisq0'], max_buffers=2, leaky=2)
                        if self.use_safe_cuda_brightness:
                            self.elements['hicon_melting_0'] = Gst.ElementFactory.make(
                                "hicon_melting_detect", "hicon-melting-0"
                            )
                            if self.elements['hicon_melting_0']:
                                self.elements['hicon_melting_0'].set_property(
                                    'config-ini', self.stream0_melting_config_ini
                                )
                                try:
                                    tapping_zones = self.elements['hicon_melting_0'].get_property(
                                        'tapping-zone-count'
                                    )
                                    deslagging_zones = self.elements['hicon_melting_0'].get_property(
                                        'deslagging-zone-count'
                                    )
                                    spectro_zones = self.elements['hicon_melting_0'].get_property(
                                        'spectro-zone-count'
                                    )
                                    logger.info(
                                        "Stream 0: C++ melting config applied "
                                        "(len=%d, zones=%s/%s/%s)",
                                        len(self.stream0_melting_config_ini),
                                        tapping_zones,
                                        deslagging_zones,
                                        spectro_zones,
                                    )
                                except Exception:
                                    logger.exception(
                                        "Stream 0: Failed to read melting plugin zone counts "
                                        "after config-ini apply"
                                    )
                                logger.info(
                                    "Stream 0: C++ melting plugin created (hicon_melting_detect)"
                                )
                            else:
                                logger.error(
                                    "Stream 0: Failed to create hicon_melting_detect element"
                                )
                        self.elements['analysis_sink0'] = Gst.ElementFactory.make(
                            "fakesink", "analysis-sink0"
                        )
                        self.elements['analysis_sink0'].set_property('sync', False)
                        self.elements['analysis_sink0'].set_property('async', False)
                        logger.info(
                            "Stream 0: decoupled analysis mode — NV12 tee "
                            "(display NV12 → nvosd_0, leaky NV12 analysis branch)"
                        )
                    else:
                        logger.info(
                            "Stream 0: decoupled analysis mode — NV12 tee "
                            "(analysis branch disabled for isolation)"
                        )
                else:
                    # Non-decoupled: single NV12→RGBA conversion, no pre-OSD tee.
                    self.elements['nvvidconv_osd_0'] = Gst.ElementFactory.make(
                        "nvvideoconvert", "nvvidconv-osd-0"
                    )
                    self._tune_stream0_postmux_convert_for_cp_plus(self.elements['nvvidconv_osd_0'])
                    self.elements['caps_osd_0'] = Gst.ElementFactory.make("capsfilter", "caps-osd-0")
                    self.elements['caps_osd_0'].set_property(
                        'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                    )
                    self.elements['preosdq0'] = Gst.ElementFactory.make("queue", "preosdq0")
                    self._configure_leaky_queue(self.elements['preosdq0'])
                    logger.info(
                        "Stream 0 (CP Plus): post-mux isolation queues enabled "
                        "(postmuxq0, preosdq0, leaky=2, max-size-buffers=16)"
                    )
                if self.stream0_bypass_pgie:
                    logger.warning("Stream 0 (CP Plus): bypassing pgie_pouring and tracker_0 for diagnostic run")
                elif self.stream0_bypass_tracker:
                    logger.warning("Stream 0 (CP Plus): bypassing tracker_0 for diagnostic run")

                # Optional annotated-output split point (post-OSD frames with overlays)
                if self.stream0_annotated_tee_enabled:
                    # Normalize OSD output caps before tee to avoid downstream caps quirks.
                    self.elements['post_osd_conv_0'] = Gst.ElementFactory.make("nvvideoconvert", "post-osd-conv-0")
                    self.elements['post_osd_caps_0'] = Gst.ElementFactory.make("capsfilter", "post-osd-caps-0")
                    if self.elements['post_osd_caps_0']:
                        self.elements['post_osd_caps_0'].set_property(
                            'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                        )
                    self.elements['tee_0'] = Gst.ElementFactory.make("tee", "tee-0")
                    self.elements['queue_display_0'] = Gst.ElementFactory.make("queue", "queue-display-0")
                    if self.elements['queue_display_0']:
                        self.elements['queue_display_0'].set_property('leaky', 2)
                        self.elements['queue_display_0'].set_property('max-size-buffers', 8)

                self.elements['sink_0'] = Gst.ElementFactory.make("fakesink", "sink-0")
                self.elements['sink_0'].set_property('sync', 0)
                self.elements['sink_0'].set_property('async', False)

        # === STREAM 1: Pyrometer Camera ===
        if 1 in self.enabled_streams:
            if self.use_nvurisrcbin_1:
                if not self._create_nvurisrcbin_chain(1, self.config['rtsp_stream_1']):
                    return False
            else:
                self._create_decode_chain(1, self.config['rtsp_stream_1'])

            self.elements['mux_1'] = self._create_streammux("mux-1")

            # Pyrometer inference (GIE-2)
            self.elements['pgie_pyrometer'] = Gst.ElementFactory.make("nvinfer", "pgie-pyrometer")
            self.elements['pgie_pyrometer'].set_property('config-file-path', self.config['config_pyrometer'])
            logger.info("Stream 1: Pyrometer nvinfer created (GIE-2)")

            # OSD for pyrometer
            self.elements['nvvidconv_osd_1'] = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-osd-1")
            self.elements['caps_osd_1'] = Gst.ElementFactory.make("capsfilter", "caps-osd-1")
            self.elements['caps_osd_1'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
            )
            self.elements['nvosd_1'] = Gst.ElementFactory.make("nvdsosd", "nvosd-1")
            if self.elements['nvosd_1']:
                self.elements['nvosd_1'].set_property('process-mode', _NVDSOSD_MODE_CPU)

            # Optional DS-native recording split point for stream 1
            if self.enable_inference_video_stream_1:
                self.elements['post_osd_conv_1'] = Gst.ElementFactory.make("nvvideoconvert", "post-osd-conv-1")
                self.elements['post_osd_caps_1'] = Gst.ElementFactory.make("capsfilter", "post-osd-caps-1")
                if self.elements['post_osd_caps_1']:
                    self.elements['post_osd_caps_1'].set_property(
                        'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                    )
                self.elements['tee_1'] = Gst.ElementFactory.make("tee", "tee-1")
                self.elements['queue_display_1'] = Gst.ElementFactory.make("queue", "queue-display-1")
                if self.elements['queue_display_1']:
                    self.elements['queue_display_1'].set_property('leaky', 2)
                    self.elements['queue_display_1'].set_property('max-size-buffers', 8)

            self.elements['sink_1'] = Gst.ElementFactory.make("fakesink", "sink-1")
            self.elements['sink_1'].set_property('sync', 0)
            self.elements['sink_1'].set_property('async', False)

        # === STREAM 2: Second Pouring Camera (pouring only, no brightness) ===
        if 2 in self.enabled_streams:
            if self.use_segment_buffer_2:
                if not self._create_segment_buffer_chain(
                    2,
                    self.config['rtsp_stream_2'],
                    self.segment_buffer_dir_2,
                    self.segment_buffer_segment_sec_2,
                    self.segment_buffer_delay_sec_2,
                    self.segment_buffer_retention_sec_2,
                ):
                    return False
            elif self.use_udp_loopback_2:
                port = self.config.get('udp_loopback_port_2', 5002)
                if not self._create_udp_loopback_chain(2, self.config['rtsp_stream_2'], port):
                    return False
            elif self.use_ffmpeg_src_2:
                if not self._create_ffmpeg_chain(2, self.config['rtsp_stream_2']):
                    return False
            elif self.use_nvurisrcbin_2:
                if not self._create_nvurisrcbin_chain(2, self.config['rtsp_stream_2']):
                    return False
            else:
                self._create_decode_chain(2, self.config['rtsp_stream_2'])

            self.elements['mux_2'] = self._create_streammux("mux-2", width=1280, height=720)

            # Pouring inference (GIE-3)
            self.elements['pgie_pouring_2'] = Gst.ElementFactory.make("nvinfer", "pgie-pouring-2")
            self.elements['pgie_pouring_2'].set_property(
                'config-file-path', self.config['config_pouring_2']
            )
            logger.info("Stream 2: Pouring nvinfer created (GIE-3)")

            # Tracker for stream 2 pouring
            self.elements['tracker_2'] = Gst.ElementFactory.make("nvtracker", "tracker-2")
            self.elements['tracker_2'].set_property('ll-lib-file', self.config['tracker_lib'])
            self.elements['tracker_2'].set_property('ll-config-file', self.config['tracker_config'])
            self.elements['tracker_2'].set_property('tracker-width', 640)
            self.elements['tracker_2'].set_property('tracker-height', 384)

            # OSD for stream 2
            self.elements['nvvidconv_osd_2'] = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-osd-2")
            self.elements['caps_osd_2'] = Gst.ElementFactory.make("capsfilter", "caps-osd-2")
            if self.elements['caps_osd_2']:
                self.elements['caps_osd_2'].set_property(
                    'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                )
            self.elements['nvosd_2'] = Gst.ElementFactory.make("nvdsosd", "nvosd-2")
            if self.elements['nvosd_2']:
                self.elements['nvosd_2'].set_property('process-mode', _NVDSOSD_MODE_CPU)

            # Optional DS-native recording split point for stream 2
            if self.enable_inference_video_stream_2:
                self.elements['post_osd_conv_2'] = Gst.ElementFactory.make("nvvideoconvert", "post-osd-conv-2")
                self.elements['post_osd_caps_2'] = Gst.ElementFactory.make("capsfilter", "post-osd-caps-2")
                if self.elements['post_osd_caps_2']:
                    self.elements['post_osd_caps_2'].set_property(
                        'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
                    )
                self.elements['tee_2'] = Gst.ElementFactory.make("tee", "tee-2")
                self.elements['queue_display_2'] = Gst.ElementFactory.make("queue", "queue-display-2")
                if self.elements['queue_display_2']:
                    self.elements['queue_display_2'].set_property('leaky', 2)
                    self.elements['queue_display_2'].set_property('max-size-buffers', 8)

            self.elements['sink_2'] = Gst.ElementFactory.make("fakesink", "sink-2")
            if self.elements['sink_2']:
                self.elements['sink_2'].set_property('sync', 0)
                self.elements['sink_2'].set_property('async', False)

        # Add all elements to pipeline
        for name, element in self.elements.items():
            if element and not name.startswith('source'):
                # rtspsrc added separately (pad-added linking)
                self.pipeline.add(element)
            elif element and name.startswith('source'):
                self.pipeline.add(element)

        # Verify all elements were created
        for name, element in self.elements.items():
            if element is None:
                logger.error(f"Failed to create element: {name}")
                return False

        logger.info("All elements created (3-stream HiCon)")
        return True

    def _is_ffmpeg_stream(self, stream_id):
        """Check if a stream uses ffmpeg pipe source (static pads, no pad-added callback)."""
        if stream_id == 0:
            return self.use_ffmpeg_src_0
        if stream_id == 2:
            return self.use_ffmpeg_src_2
        return False

    def _is_segment_buffer_stream(self, stream_id):
        """Check if a stream uses delayed segment-buffer ingest."""
        if stream_id == 0:
            return self.use_segment_buffer_0
        if stream_id == 2:
            return self.use_segment_buffer_2
        return False

    def _is_udp_loopback_stream(self, stream_id):
        """Check if a stream uses UDP loopback source (tsdemux dynamic pad)."""
        if stream_id == 0:
            return self.use_udp_loopback_0
        if stream_id == 2:
            return self.use_udp_loopback_2
        return False

    def _is_tsdemux_stream(self, stream_id):
        """Check if a stream reaches the parser via tsdemux (UDP loopback only; segment buffer now uses static H264 chain)."""
        return self._is_udp_loopback_stream(stream_id)

    def _link_decode_chain(self, stream_id):
        """Link decode chain, with Stream 0 queue isolation before depay and before mux."""
        sid = str(stream_id)
        if (stream_id == 0 and self.use_nvurisrcbin_0) or \
           (stream_id == 1 and self.use_nvurisrcbin_1) or \
           (stream_id == 2 and self.use_nvurisrcbin_2):
            # nvurisrcbin handles decode internally — nothing to link here.
            # The vsrc pad is linked to premuxq/mux via pad-added callback.
            return True
        if self._is_segment_buffer_stream(stream_id):
            # Segment buffer uses raw H264 files — static chain, same as ffmpeg pipe path.
            # No tsdemux/tsparse dynamic pad deadlock.
            chain = [
                (f'source{sid}', f'segbufq{sid}'),
                (f'segbufq{sid}', f'parser{sid}'),
                (f'parser{sid}', f'vidcaps{sid}'),
                (f'vidcaps{sid}', f'decoder{sid}'),
                (f'decoder{sid}', f'nvvidconv{sid}'),
                (f'nvvidconv{sid}', f'caps{sid}'),
            ]
            if stream_id == 0:
                chain.append(('caps0', 'premuxq0'))
            for src_name, dst_name in chain:
                if not self.elements[src_name].link(self.elements[dst_name]):
                    logger.error(f"Failed to link {src_name} -> {dst_name}")
                    return False
            return True
        if self._is_udp_loopback_stream(stream_id):
            # UDP loopback: udpsrc → tsdemux (static link), then tsdemux pad-added → demuxq → parser...
            if not self.elements[f'source{sid}'].link(self.elements[f'tsdemux{sid}']):
                logger.error(f"Failed to link source{sid} -> tsdemux{sid}")
                return False
            return True
        if self._is_ffmpeg_stream(stream_id):
            # ffmpeg/fdsrc has static pads — full chain linked statically (no depay)
            # ffmpegq decouples pipe read from downstream to prevent back-pressure
            chain = [
                (f'source{sid}', f'ffmpegq{sid}'),
                (f'ffmpegq{sid}', f'parser{sid}'),
                (f'parser{sid}', f'vidcaps{sid}'),
                (f'vidcaps{sid}', f'decoder{sid}'),
                (f'decoder{sid}', f'nvvidconv{sid}'),
                (f'nvvidconv{sid}', f'caps{sid}'),
            ]
            if stream_id == 0:
                chain.append(('caps0', 'premuxq0'))
        elif stream_id == 0:
            chain = [
                ('srcq0', 'depay0'),
                ('depay0', 'parser0'),
                ('parser0', 'vidcaps0'),
                ('vidcaps0', 'decoder0'),
                ('decoder0', 'nvvidconv0'),
                ('nvvidconv0', 'caps0'),
                ('caps0', 'premuxq0'),
            ]
        else:
            chain = [
                (f'depay{sid}', f'parser{sid}'),
                (f'parser{sid}', f'vidcaps{sid}'),
                (f'vidcaps{sid}', f'decoder{sid}'),
                (f'decoder{sid}', f'nvvidconv{sid}'),
                (f'nvvidconv{sid}', f'caps{sid}'),
            ]
        for src_name, dst_name in chain:
            if not self.elements[src_name].link(self.elements[dst_name]):
                logger.error(f"Failed to link {src_name} -> {dst_name}")
                return False
        return True

    def _link_to_mux(self, src_name, mux_name):
        """Link source element src pad to mux sink_0 pad."""
        sinkpad = self.elements[mux_name].request_pad_simple("sink_0")
        srcpad = self.elements[src_name].get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error(f"Failed to link {src_name} -> {mux_name}.sink_0")
            return False
        return True

    def _link_tee_src_to_element(self, tee_name, dst_name):
        """Link one tee src pad to destination sink pad."""
        tee_pad = self.elements[tee_name].request_pad_simple("src_%u")
        dst_pad = self.elements[dst_name].get_static_pad("sink")
        if not tee_pad or not dst_pad:
            logger.error(f"Failed to get pads for {tee_name} -> {dst_name}")
            return False
        if tee_pad.link(dst_pad) != Gst.PadLinkReturn.OK:
            logger.error(f"Failed to link tee pad {tee_name} -> {dst_name}")
            return False
        return True

    def _link_all_branches(self):
        """Link all pipeline branches."""

        # === Stream 0: Process Camera ===
        if 0 in self.enabled_streams:
            if not self._link_decode_chain(0):
                return False
            if self.stream0_decode_only_mode:
                if not self.elements['premuxq0'].link(self.elements['decode_sink_0']):
                    logger.error("Failed to link premuxq0 -> decode_sink_0")
                    return False
                logger.info("Stream 0: decode-only diagnostic chain linked")
            elif self.stream0_postconv_only_mode:
                if not self._link_to_mux('premuxq0', 'mux_0'):
                    return False
                if not self.elements['mux_0'].link(self.elements['postmuxq0']):
                    logger.error("Failed to link mux_0 -> postmuxq0")
                    return False
                if not self.elements['postmuxq0'].link(self.elements['nvvidconv_osd_0']):
                    logger.error("Failed to link postmuxq0 -> nvvidconv_osd_0")
                    return False
                if not self.elements['nvvidconv_osd_0'].link(self.elements['postconv_sink_0']):
                    logger.error("Failed to link nvvidconv_osd_0 -> postconv_sink_0")
                    return False
                logger.info("Stream 0: post-convert-only diagnostic chain linked")
            elif self.stream0_preosd_only_mode:
                if not self._link_to_mux('premuxq0', 'mux_0'):
                    return False
                if not self.elements['mux_0'].link(self.elements['postmuxq0']):
                    logger.error("Failed to link mux_0 -> postmuxq0")
                    return False
                if not self.elements['postmuxq0'].link(self.elements['nvvidconv_osd_0']):
                    logger.error("Failed to link postmuxq0 -> nvvidconv_osd_0")
                    return False
                if not self.elements['nvvidconv_osd_0'].link(self.elements['caps_osd_0']):
                    logger.error("Failed to link nvvidconv_osd_0 -> caps_osd_0")
                    return False
                if not self.elements['caps_osd_0'].link(self.elements['preosdq0']):
                    logger.error("Failed to link caps_osd_0 -> preosdq0")
                    return False
                if not self.elements['preosdq0'].link(self.elements['preosd_sink_0']):
                    logger.error("Failed to link preosdq0 -> preosd_sink_0")
                    return False
                logger.info("Stream 0: pre-OSD-only diagnostic chain linked")
            elif self.stream0_postmux_only_mode:
                if not self._link_to_mux('premuxq0', 'mux_0'):
                    return False
                if not self.elements['mux_0'].link(self.elements['postmuxq0']):
                    logger.error("Failed to link mux_0 -> postmuxq0")
                    return False
                if not self.elements['postmuxq0'].link(self.elements['postmux_sink_0']):
                    logger.error("Failed to link postmuxq0 -> postmux_sink_0")
                    return False
                logger.info("Stream 0: post-mux-only diagnostic chain linked")
            else:
                if not self._link_to_mux('premuxq0', 'mux_0'):
                    return False

                # mux_0 → postmuxq0 → pouring → tracker → nvvidconv → caps_rgba → preosdq0 → osd
                chain_0 = [('mux_0', 'postmuxq0')]
                stream0_head = 'postmuxq0'
                if not self.stream0_bypass_pgie:
                    chain_0.append((stream0_head, 'pgie_pouring'))
                    stream0_head = 'pgie_pouring'
                if not self.stream0_bypass_tracker:
                    chain_0.append((stream0_head, 'tracker_0'))
                    stream0_head = 'tracker_0'
                if self.stream0_decoupled_analysis_mode:
                    chain_0.append((stream0_head, 'tee_stream0_analysis'))
                else:
                    chain_0.extend([
                        (stream0_head, 'nvvidconv_osd_0'),
                        ('nvvidconv_osd_0', 'caps_osd_0'),
                        ('caps_osd_0', 'preosdq0'),
                        ('preosdq0', 'nvosd_0'),
                    ])
                for src_name, dst_name in chain_0:
                    if not self.elements[src_name].link(self.elements[dst_name]):
                        logger.error(f"Failed to link {src_name} -> {dst_name}")
                        return False
                if self.stream0_decoupled_analysis_mode:
                    if not self._link_tee_src_to_element('tee_stream0_analysis', 'displayq0'):
                        return False
                    display_chain = []
                    display_head = 'displayq0'
                    display_chain.append((display_head, 'nvosd_0'))
                    for src_name, dst_name in display_chain:
                        if not self.elements[src_name].link(self.elements[dst_name]):
                            logger.error(f"Failed to link {src_name} -> {dst_name}")
                            return False
                    if self.stream0_analysis_branch_enabled:
                        if not self._link_tee_src_to_element('tee_stream0_analysis', 'analysisq0'):
                            return False
                        analysis_chain = []
                        analysis_head = 'analysisq0'
                        if 'hicon_melting_0' in self.elements and self.elements.get('hicon_melting_0'):
                            analysis_chain.append((analysis_head, 'hicon_melting_0'))
                            analysis_head = 'hicon_melting_0'
                            logger.info(
                                "Stream 0: C++ melting plugin placed on analysis branch "
                                "(metadata-only CUDA path)"
                            )
                        analysis_chain.append((analysis_head, 'analysis_sink0'))
                        for src_name, dst_name in analysis_chain:
                            if not self.elements[src_name].link(self.elements[dst_name]):
                                logger.error(f"Failed to link {src_name} -> {dst_name}")
                                return False
                    else:
                        logger.info(
                            "Stream 0: analysis branch omitted for isolation; only display path "
                            "linked from tee_stream0_analysis"
                        )

                if self.stream0_annotated_tee_enabled:
                    # Split annotated stream: display path + optional recording/relay branches
                    if not self.elements['nvosd_0'].link(self.elements['post_osd_conv_0']):
                        logger.error("Failed to link nvosd_0 -> post_osd_conv_0")
                        return False
                    if not self.elements['post_osd_conv_0'].link(self.elements['post_osd_caps_0']):
                        logger.error("Failed to link post_osd_conv_0 -> post_osd_caps_0")
                        return False
                    if not self.elements['post_osd_caps_0'].link(self.elements['tee_0']):
                        logger.error("Failed to link post_osd_caps_0 -> tee_0")
                        return False
                    if not self._link_tee_src_to_element('tee_0', 'queue_display_0'):
                        return False
                    if not self.elements['queue_display_0'].link(self.elements['sink_0']):
                        logger.error("Failed to link queue_display_0 -> sink_0")
                        return False
                else:
                    if not self.elements['nvosd_0'].link(self.elements['sink_0']):
                        logger.error("Failed to link nvosd_0 -> sink_0")
                        return False
                logger.info("Stream 0: Process camera chain linked")

        # === Stream 1: Pyrometer Camera ===
        if 1 in self.enabled_streams:
            if not self._link_decode_chain(1):
                return False
            if not self.use_nvurisrcbin_1:
                if not self._link_to_mux('caps1', 'mux_1'):
                    return False
            else:
                # nvurisrcbin: pad-added links vsrc → premuxq1; link premuxq1 → mux_1 here
                if not self._link_to_mux('premuxq1', 'mux_1'):
                    return False

            # mux_1 → pyrometer → nvvidconv → caps_rgba → osd → [tee_1 →] sink_1
            chain_1 = [
                ('mux_1', 'pgie_pyrometer'),
                ('pgie_pyrometer', 'nvvidconv_osd_1'),
                ('nvvidconv_osd_1', 'caps_osd_1'),
                ('caps_osd_1', 'nvosd_1'),
            ]
            for src_name, dst_name in chain_1:
                if not self.elements[src_name].link(self.elements[dst_name]):
                    logger.error(f"Failed to link {src_name} -> {dst_name}")
                    return False

            if self.enable_inference_video_stream_1:
                if not self.elements['nvosd_1'].link(self.elements['post_osd_conv_1']):
                    logger.error("Failed to link nvosd_1 -> post_osd_conv_1")
                    return False
                if not self.elements['post_osd_conv_1'].link(self.elements['post_osd_caps_1']):
                    logger.error("Failed to link post_osd_conv_1 -> post_osd_caps_1")
                    return False
                if not self.elements['post_osd_caps_1'].link(self.elements['tee_1']):
                    logger.error("Failed to link post_osd_caps_1 -> tee_1")
                    return False
                if not self._link_tee_src_to_element('tee_1', 'queue_display_1'):
                    return False
                if not self.elements['queue_display_1'].link(self.elements['sink_1']):
                    logger.error("Failed to link queue_display_1 -> sink_1")
                    return False
            else:
                if not self.elements['nvosd_1'].link(self.elements['sink_1']):
                    logger.error("Failed to link nvosd_1 -> sink_1")
                    return False
            logger.info("Stream 1: Pyrometer camera chain linked")

        # === Stream 2: Second Pouring Camera ===
        if 2 in self.enabled_streams:
            if not self._link_decode_chain(2):
                return False
            if not self.use_nvurisrcbin_2:
                if not self._link_to_mux('caps2', 'mux_2'):
                    return False
            else:
                # nvurisrcbin: pad-added links vsrc → premuxq2; link premuxq2 → mux_2 here
                if not self._link_to_mux('premuxq2', 'mux_2'):
                    return False

            # mux_2 → [pouring(GIE-3) → tracker →] [cpp pouring →] nvvidconv → caps_rgba → osd → sink
            chain_2 = []
            stream2_head = 'mux_2'
            chain_2.append(('mux_2', 'pgie_pouring_2'))
            chain_2.append(('pgie_pouring_2', 'tracker_2'))
            stream2_head = 'tracker_2'
            chain_2.append((stream2_head, 'nvvidconv_osd_2'))
            chain_2.extend([
                ('nvvidconv_osd_2', 'caps_osd_2'),
                ('caps_osd_2', 'nvosd_2'),
            ])
            if self.enable_inference_video_stream_2 and 'tee_2' in self.elements and self.elements.get('tee_2'):
                chain_2.extend([
                    ('nvosd_2', 'post_osd_conv_2'),
                    ('post_osd_conv_2', 'post_osd_caps_2'),
                    ('post_osd_caps_2', 'tee_2'),
                ])
            else:
                chain_2.append(('nvosd_2', 'sink_2'))
            for src_name, dst_name in chain_2:
                if not self.elements[src_name].link(self.elements[dst_name]):
                    logger.error(f"Failed to link {src_name} -> {dst_name}")
                    return False
            # Link tee_2 display branch if recording is active
            if self.enable_inference_video_stream_2 and 'tee_2' in self.elements and self.elements.get('tee_2'):
                if not self._link_tee_src_to_element('tee_2', 'queue_display_2'):
                    return False
                if not self.elements['queue_display_2'].link(self.elements['sink_2']):
                    logger.error("Failed to link queue_display_2 -> sink_2")
                    return False
            logger.info("Stream 2: Second pouring camera chain linked")

        # Connect pad-added callbacks for RTSP sources
        # (ffmpeg/fdsrc and segment-buffer both use static pads — no callback needed)
        for i in self.enabled_streams:
            if self._is_udp_loopback_stream(i):
                # tsdemux has dynamic pads — connect pad-added to link to demuxq → parser chain
                self.elements[f'tsdemux{i}'].connect("pad-added", self._cb_tsdemux_pad_added, i)
            elif self._is_ffmpeg_stream(i) or self._is_segment_buffer_stream(i):
                pass  # static src pad, already fully linked in _link_decode_chain
            elif (i == 0 and self.use_nvurisrcbin_0) or \
                 (i == 1 and self.use_nvurisrcbin_1) or \
                 (i == 2 and self.use_nvurisrcbin_2):
                self.elements[f'source{i}'].connect(
                    "pad-added", self._cb_nvurisrcbin_pad_added, i
                )
            else:
                self.elements[f'source{i}'].connect("pad-added", self._cb_newpad, i)

        return True

    # Known audio RTP payload names — silently ignored (we only decode video)
    _AUDIO_ENCODINGS = frozenset([
        'PCMU', 'PCMA', 'OPUS', 'G722', 'G726-16', 'G726-24', 'G726-32', 'G726-40',
        'MPEG4-GENERIC', 'MP4A-LATM', 'AC3', 'AAC',
    ])

    def _cb_newpad(self, decodebin, pad, stream_id):
        """RTSP pad-added callback to link dynamic pads (supports H264 and H265).

        Audio pads (PCMA, PCMU, OPUS, etc.) are silently skipped — we only want video.
        The Python GI bindings raise Gst.LinkError on pad.link() failure instead of
        returning an error code, so the link attempt is wrapped in try/except.
        """
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return

        structure = caps.get_structure(0)
        if not structure.get_name().startswith("application/x-rtp"):
            return

        encoding_name = (structure.get_string("encoding-name") or "").upper()

        # Audio tracks must be consumed (not silently dropped) — rtspsrc already did
        # SETUP/PLAY for the audio RTSP session before this callback fires. Leaving
        # the pad unlinked causes RTP buffer buildup and RTCP Receiver Report starvation,
        # which makes the camera time out and drop the entire connection (~4-5 min).
        if encoding_name in self._AUDIO_ENCODINGS:
            audio_sink = Gst.ElementFactory.make(
                "fakesink", f"audio-discard-{stream_id}-{id(pad)}"
            )
            if audio_sink:
                audio_sink.set_property('sync', False)
                audio_sink.set_property('async', False)
                self.pipeline.add(audio_sink)
                audio_sink.sync_state_with_parent()
                try:
                    pad.link(audio_sink.get_static_pad("sink"))
                    logger.debug(
                        f"Stream {stream_id}: audio ({encoding_name}) → discard sink"
                    )
                except Exception:
                    pass
            return

        expected_codec = str(self.config.get(f'rtsp_codec_{stream_id}', 'h265')).upper()
        expected_encoding = "H265" if expected_codec == "H265" else "H264"

        # Silently skip non-video tracks (ONVIF metadata, unknown encodings)
        if encoding_name != expected_encoding:
            return

        target_name = 'srcq0' if stream_id == 0 and 'srcq0' in self.elements else f'depay{stream_id}'
        target_sinkpad = self.elements[target_name].get_static_pad("sink")
        if not target_sinkpad.is_linked():
            try:
                pad.link(target_sinkpad)
                # Block EOS events from rtpsession timeout on TCP streams.
                # GStreamer's rtpsession declares "source timed out" when RTCP
                # Sender Reports are missing and pushes EOS, even though data is
                # still flowing over TCP.  Dropping this EOS keeps the stream alive;
                # real connection failures are caught by bus ERROR messages and the
                # FPS watchdog.
                stream_protocol = str(self.config.get(
                    f'rtsp_protocol_{stream_id}', 'auto'
                ) or 'auto').lower()
                if stream_protocol == 'tcp':
                    pad.add_probe(
                        Gst.PadProbeType.EVENT_DOWNSTREAM,
                        self._eos_drop_probe, stream_id,
                    )
                logger.info(
                    f"Stream {stream_id}: RTSP pad linked "
                    f"(encoding={encoding_name}, codec={expected_codec})"
                )
            except Exception as exc:
                logger.error(
                    f"Stream {stream_id}: RTSP pad link failed "
                    f"(encoding={encoding_name}): {exc}"
                )

    def _cb_nvurisrcbin_pad_added(self, uribin, pad, stream_id):
        """nvurisrcbin pad-added callback — link video src pad to premuxq0."""
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return

        name = pad.get_name()
        # nvurisrcbin emits vsrc_%u for video, asrc_%u for audio
        if not name.startswith("vsrc_"):
            logger.debug(f"Stream {stream_id}: nvurisrcbin ignoring non-video pad {name}")
            return

        # Link to premuxq (leaky queue) for ALL streams — absorbs backpressure
        target_name = f'premuxq{stream_id}'
        target_pad = self.elements[target_name].get_static_pad("sink")
        if target_pad.is_linked():
            logger.debug(f"Stream {stream_id}: nvurisrcbin {target_name} already linked")
            return

        try:
            pad.link(target_pad)
            logger.info(
                f"Stream {stream_id}: nvurisrcbin video pad '{name}' linked to {target_name}"
            )
        except Exception as exc:
            logger.error(
                f"Stream {stream_id}: nvurisrcbin pad link failed: {exc}"
            )

    def _cb_tsdemux_pad_added(self, tsdemux, pad, stream_id):
        """tsdemux pad-added callback — link video pad to demuxq → parser → ... → mux chain."""
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            return

        struct = caps.get_structure(0)
        name = struct.get_name()
        # Only link video pads (video/x-h264 or video/x-h265); skip audio
        if not name.startswith("video/"):
            logger.debug(f"Stream {stream_id}: tsdemux ignoring non-video pad ({name})")
            return

        sid = str(stream_id)
        demuxq = self.elements.get(f'demuxq{sid}')
        if demuxq is None:
            logger.error(f"Stream {stream_id}: demuxq{sid} not found")
            return

        sink_pad = demuxq.get_static_pad("sink")
        if sink_pad.is_linked():
            logger.debug(f"Stream {stream_id}: tsdemux video pad already linked")
            return

        try:
            pad.link(sink_pad)
            logger.info(f"Stream {stream_id}: tsdemux video pad linked to demuxq{sid}")
        except Exception as exc:
            logger.error(f"Stream {stream_id}: tsdemux pad link failed: {exc}")
            return

        # Now statically link the rest of the chain
        chain = [
            (f'demuxq{sid}', f'parser{sid}'),
            (f'parser{sid}', f'vidcaps{sid}'),
            (f'vidcaps{sid}', f'decoder{sid}'),
            (f'decoder{sid}', f'nvvidconv{sid}'),
            (f'nvvidconv{sid}', f'caps{sid}'),
        ]
        if stream_id == 0:
            chain.append(('caps0', 'premuxq0'))
        for src_name, dst_name in chain:
            el_src = self.elements.get(src_name)
            el_dst = self.elements.get(dst_name)
            if el_src is None or el_dst is None:
                logger.error(f"Stream {stream_id}: missing element {src_name} or {dst_name}")
                return
            el_src.sync_state_with_parent()
            el_dst.sync_state_with_parent()
            if not el_src.link(el_dst):
                logger.error(f"Stream {stream_id}: failed to link {src_name} -> {dst_name}")
                return
        if self._is_segment_buffer_stream(stream_id):
            logger.info(f"Stream {stream_id}: segment buffer chain fully linked via tsdemux pad-added")
        else:
            logger.info(f"Stream {stream_id}: UDP loopback chain fully linked via tsdemux pad-added")

    def _configure_rtsp_source(self, source, location, stream_id):
        """Configure RTSP source transport, timeouts, and reconnection behavior."""
        if not location:
            return

        protocol = str(self.config.get(f'rtsp_protocol_{stream_id}', 'auto') or 'auto').lower()
        if protocol not in {'auto', 'tcp', 'udp'}:
            logger.warning(
                f"Stream {stream_id}: invalid RTSP protocol {protocol!r}; using auto"
            )
            protocol = 'auto'

        source.set_property('location', location)
        latency_ms = int(self.config.get('rtsp_latency_ms', 4000) or 4000)
        source.set_property('latency', latency_ms)
        source.set_property('drop-on-latency', True)
        source.set_property('buffer-mode', 0)  # none: raw RTP timestamps (longest proven runtime)
        source.set_property('do-rtsp-keep-alive', True)

        if protocol == 'tcp':
            self._set_rtsp_protocol(source, GstRtsp.RTSPLowerTrans.TCP, 'tcp', stream_id)
        elif protocol == 'udp':
            self._set_rtsp_protocol(source, GstRtsp.RTSPLowerTrans.UDP, 'udp', stream_id)

        rtsp_port_retry = int(self.config.get('rtsp_port_retry', 0) or 0)
        if rtsp_port_retry > 0:
            self._set_rtsp_property(source, 'retry', rtsp_port_retry, stream_id)

        rtsp_udp_timeout_us = int(self.config.get('rtsp_udp_timeout_us', 0) or 0)
        effective_udp_timeout_us = 0
        if protocol in {'auto', 'udp'} and rtsp_udp_timeout_us > 0:
            self._set_rtsp_property(source, 'timeout', rtsp_udp_timeout_us, stream_id)
            effective_udp_timeout_us = rtsp_udp_timeout_us

        rtsp_tcp_timeout_us = int(self.config.get('rtsp_tcp_timeout_us', 0) or 0)
        effective_tcp_timeout_us = 0
        if protocol in {'auto', 'tcp'} and rtsp_tcp_timeout_us > 0:
            self._set_rtsp_property(source, 'tcp-timeout', rtsp_tcp_timeout_us, stream_id)
            effective_tcp_timeout_us = rtsp_tcp_timeout_us

        rtsp_do_retransmission = self.config.get('rtsp_do_retransmission')
        if rtsp_do_retransmission is not None:
            rtsp_do_retransmission = bool(rtsp_do_retransmission)
            self._set_rtsp_property(
                source, 'do-retransmission',
                rtsp_do_retransmission,
                stream_id
            )
        else:
            rtsp_do_retransmission = True

        # Connect to new-manager signal to configure rtpbin and prevent
        # RTP session timeout caused by missing RTCP Sender Reports from NVR.
        # Root cause: GStreamer's rtpsession (inside rtpbin) declares the RTP
        # source "timed out" when no RTCP SR is received, killing the stream.
        # ffmpeg doesn't have this issue because it reads RTP directly without
        # a separate RTP session manager.
        source.connect('new-manager', self._on_rtpbin_created, stream_id)

        logger.info(
            f"Stream {stream_id}: RTSP config "
            f"protocol={protocol}, latency={latency_ms}ms, drop-on-latency=True, buffer-mode=0(none), "
            f"timeout={effective_udp_timeout_us}us, tcp-timeout={effective_tcp_timeout_us}us, "
            f"retry={rtsp_port_retry}, do-retransmission={rtsp_do_retransmission}"
        )

    @staticmethod
    def _eos_drop_probe(pad, info, stream_id):
        """Drop EOS events from rtspsrc to prevent rtpsession false timeout.

        In TCP interleaved mode, rtpsession declares 'source timed out' when
        RTCP Sender Reports are missing from the NVR and pushes EOS, even
        though RTP data is still flowing.  Dropping this EOS keeps the stream
        alive.  Real failures are caught by bus ERROR messages and FPS watchdog.
        """
        event = info.get_event()
        if event and event.type == Gst.EventType.EOS:
            logger.warning(
                f"Stream {stream_id}: EOS dropped on rtspsrc pad "
                f"(rtpsession false timeout — stream continues)"
            )
            return Gst.PadProbeReturn.DROP
        return Gst.PadProbeReturn.OK

    @staticmethod
    def _on_rtpbin_created(rtspsrc, manager, stream_id):
        """Configure rtpbin to prevent premature RTP session timeout.

        The Hikvision NVR may not send RTCP Sender Reports frequently enough
        (or at all) over TCP interleaved mode, causing GStreamer's rtpsession
        to declare the source dead.  We set a very long RTCP min-interval via
        the on-new-ssrc signal so the timeout (5× interval) is effectively
        infinite.
        """
        try:
            manager.set_property('ntp-sync', False)
            manager.set_property('ntp-time-source', 3)  # NTP_TIME_SOURCE_CLOCK_TIME
            manager.set_property('max-rtcp-rtp-time-diff', -1)  # -1 = disabled
            # When a new SSRC is seen, set rtcp-min-interval on the internal
            # RTPSession to 1 hour (in ns).  Source timeout = 5 × interval = 5h.
            manager.connect('on-new-ssrc', DeepStreamPipelineBuilder._on_new_ssrc, stream_id)
            logger.info(
                f"Stream {stream_id}: rtpbin configured — "
                f"ntp-sync=False, ntp-time-source=3, max-rtcp-rtp-time-diff=-1, "
                f"on-new-ssrc handler attached"
            )
        except Exception as exc:
            logger.warning(f"Stream {stream_id}: failed to configure rtpbin: {exc}")

    @staticmethod
    def _on_new_ssrc(rtpbin, session_id, ssrc, stream_id):
        """Set a very long RTCP min-interval on the internal RTPSession.

        This makes the source timeout (5× RTCP interval) effectively infinite,
        preventing rtpsession from falsely declaring the NVR source as timed out.
        """
        try:
            internal_session = rtpbin.emit('get-internal-session', session_id)
            if internal_session:
                # 1 hour in nanoseconds — source timeout becomes 5 hours
                one_hour_ns = 3600 * 1_000_000_000
                internal_session.set_property('rtcp-min-interval', one_hour_ns)
                logger.info(
                    f"Stream {stream_id}: rtpsession[{session_id}] "
                    f"rtcp-min-interval set to 1h (SSRC={ssrc:#010x})"
                )
        except Exception as exc:
            logger.debug(f"Stream {stream_id}: rtpsession config failed: {exc}")

    @staticmethod
    def _set_rtsp_property(source, name, value, stream_id):
        try:
            source.set_property(name, value)
        except Exception as exc:
            logger.warning(f"Stream {stream_id}: failed to set {name}={value}: {exc}")

    @staticmethod
    def _set_rtsp_protocol(source, enum_value, text_value, stream_id):
        try:
            source.set_property('protocols', enum_value)
        except Exception as exc:
            logger.warning(
                f"Stream {stream_id}: failed to set protocols={text_value} via enum ({exc}); "
                f"retrying with string"
            )
            try:
                source.set_property('protocols', text_value)
            except Exception as inner_exc:
                logger.warning(
                    f"Stream {stream_id}: failed to set protocols={text_value} ({inner_exc})"
                )
