"""
DeepStream Pipeline Builder - HiCon 2-Stream Architecture
Constructs DS 7.1 pipeline for induction furnace monitoring:
  Stream 0 (Process Camera): pouring detection (nvinfer) + brightness analysis (probe)
  Stream 1 (Pyrometer Camera): rod detection (nvinfer)
"""
import logging
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GstRtsp

logger = logging.getLogger(__name__)


class DeepStreamPipelineBuilder:
    """
    Builds DeepStream 7.1 2-stream pipeline for HiCon furnace monitoring.

    Architecture:
    - Stream 0 → nvv4l2decoder → mux_0 → nvinfer(pouring) → nvtracker → nvosd → sink_0
    - Stream 1 → nvv4l2decoder → mux_1 → nvinfer(pyrometer) → nvosd → sink_1
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
        """
        self.config = config
        self.pipeline = None
        self.elements = {}
        self.enable_inference_video = bool(config.get('enable_inference_video', False))

    def create_pipeline(self):
        """
        Create complete 2-stream pipeline.

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

        logger.info("2-stream HiCon pipeline created successfully")
        return self.pipeline, self.elements

    def _create_streammux(self, name, batch_size=1, width=1920, height=1080):
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

    def _create_decode_chain(self, stream_id, rtsp_url):
        """Create RTSP source → depay → parse → decode → nvvidconv → caps chain."""
        sid = str(stream_id)

        self.elements[f'source{sid}'] = Gst.ElementFactory.make("rtspsrc", f"source{sid}")
        self._configure_rtsp_source(self.elements[f'source{sid}'], rtsp_url, stream_id)

        self.elements[f'depay{sid}'] = Gst.ElementFactory.make("rtph264depay", f"depay{sid}")

        self.elements[f'h264parser{sid}'] = Gst.ElementFactory.make("h264parse", f"h264parser{sid}")
        self.elements[f'h264parser{sid}'].set_property('config-interval', -1)
        self.elements[f'h264parser{sid}'].set_property('disable-passthrough', True)

        self.elements[f'h264caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"h264caps{sid}")
        self.elements[f'h264caps{sid}'].set_property(
            "caps",
            Gst.Caps.from_string("video/x-h264, stream-format=byte-stream, alignment=au")
        )

        self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("nvv4l2decoder", f"decoder{sid}")
        if self.elements[f'decoder{sid}']:
            self.elements[f'decoder{sid}'].set_property('num-extra-surfaces', 8)
            logger.info(f"Stream {sid}: Using hardware decoder (nvv4l2decoder)")
        else:
            logger.warning(f"Stream {sid}: HW decoder unavailable, using software decoder")
            self.elements[f'decoder{sid}'] = Gst.ElementFactory.make("avdec_h264", f"decoder{sid}")

        self.elements[f'nvvidconv{sid}'] = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv{sid}")
        self.elements[f'caps{sid}'] = Gst.ElementFactory.make("capsfilter", f"caps{sid}")
        self.elements[f'caps{sid}'].set_property(
            'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        )

        logger.info(f"Stream {sid}: Decode chain created")

    def _create_all_elements(self):
        """Create all pipeline elements for 2-stream HiCon."""

        # Determine enabled streams
        self.enabled_streams = []
        for i in range(2):
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
            self._create_decode_chain(0, self.config['rtsp_stream_0'])

            self.elements['mux_0'] = self._create_streammux("mux-0")

            # Pouring inference (GIE-1)
            self.elements['pgie_pouring'] = Gst.ElementFactory.make("nvinfer", "pgie-pouring")
            self.elements['pgie_pouring'].set_property('config-file-path', self.config['config_pouring'])
            logger.info("Stream 0: Pouring nvinfer created (GIE-1)")

            # Tracker for pouring
            self.elements['tracker_0'] = Gst.ElementFactory.make("nvtracker", "tracker-0")
            self.elements['tracker_0'].set_property('ll-lib-file', self.config['tracker_lib'])
            self.elements['tracker_0'].set_property('ll-config-file', self.config['tracker_config'])
            self.elements['tracker_0'].set_property('tracker-width', 640)
            self.elements['tracker_0'].set_property('tracker-height', 384)

            # OSD + convert for RGBA (needed for brightness probe)
            self.elements['nvvidconv_osd_0'] = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-osd-0")
            self.elements['caps_osd_0'] = Gst.ElementFactory.make("capsfilter", "caps-osd-0")
            self.elements['caps_osd_0'].set_property(
                'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
            )
            self.elements['nvosd_0'] = Gst.ElementFactory.make("nvdsosd", "nvosd-0")

            # Optional DS-native recording split point (post-OSD annotated frames)
            if self.enable_inference_video:
                # Normalize OSD output caps before tee to avoid downstream caps quirks.
                self.elements['post_osd_conv_0'] = Gst.ElementFactory.make("nvvideoconvert", "post-osd-conv-0")
                self.elements['post_osd_caps_0'] = Gst.ElementFactory.make("capsfilter", "post-osd-caps-0")
                if self.elements['post_osd_caps_0']:
                    self.elements['post_osd_caps_0'].set_property(
                        'caps', Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
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

            self.elements['sink_1'] = Gst.ElementFactory.make("fakesink", "sink-1")
            self.elements['sink_1'].set_property('sync', 0)
            self.elements['sink_1'].set_property('async', False)

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

        logger.info("All elements created (2-stream HiCon)")
        return True

    def _link_decode_chain(self, stream_id):
        """Link decode chain: depay → parser → caps → decoder → nvvidconv → caps."""
        sid = str(stream_id)
        chain = [
            (f'depay{sid}', f'h264parser{sid}'),
            (f'h264parser{sid}', f'h264caps{sid}'),
            (f'h264caps{sid}', f'decoder{sid}'),
            (f'decoder{sid}', f'nvvidconv{sid}'),
            (f'nvvidconv{sid}', f'caps{sid}'),
        ]
        for src_name, dst_name in chain:
            if not self.elements[src_name].link(self.elements[dst_name]):
                logger.error(f"Failed to link {src_name} -> {dst_name}")
                return False
        return True

    def _link_to_mux(self, caps_name, mux_name):
        """Link caps src pad to mux sink_0 pad."""
        sinkpad = self.elements[mux_name].request_pad_simple("sink_0")
        srcpad = self.elements[caps_name].get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            logger.error(f"Failed to link {caps_name} -> {mux_name}.sink_0")
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
            if not self._link_to_mux('caps0', 'mux_0'):
                return False

            # mux_0 → pouring → tracker → nvvidconv → caps_rgba → osd
            chain_0 = [
                ('mux_0', 'pgie_pouring'),
                ('pgie_pouring', 'tracker_0'),
                ('tracker_0', 'nvvidconv_osd_0'),
                ('nvvidconv_osd_0', 'caps_osd_0'),
                ('caps_osd_0', 'nvosd_0'),
            ]
            for src_name, dst_name in chain_0:
                if not self.elements[src_name].link(self.elements[dst_name]):
                    logger.error(f"Failed to link {src_name} -> {dst_name}")
                    return False

            if self.enable_inference_video:
                # Split annotated stream: display path + recording path (added later by RecordingManager)
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
            if not self._link_to_mux('caps1', 'mux_1'):
                return False

            # mux_1 → pyrometer → nvvidconv → caps_rgba → osd → sink
            chain_1 = [
                ('mux_1', 'pgie_pyrometer'),
                ('pgie_pyrometer', 'nvvidconv_osd_1'),
                ('nvvidconv_osd_1', 'caps_osd_1'),
                ('caps_osd_1', 'nvosd_1'),
                ('nvosd_1', 'sink_1'),
            ]
            for src_name, dst_name in chain_1:
                if not self.elements[src_name].link(self.elements[dst_name]):
                    logger.error(f"Failed to link {src_name} -> {dst_name}")
                    return False
            logger.info("Stream 1: Pyrometer camera chain linked")

        # Connect pad-added callbacks for RTSP sources
        for i in self.enabled_streams:
            self.elements[f'source{i}'].connect("pad-added", self._cb_newpad, i)

        return True

    def _cb_newpad(self, decodebin, pad, stream_id):
        """RTSP pad-added callback to link dynamic pads."""
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            logger.warning(f"RTSP pad has no caps for stream {stream_id}")
            return

        structure = caps.get_structure(0)
        caps_name = structure.get_name()

        if caps_name.startswith("application/x-rtp"):
            encoding_name = structure.get_string("encoding-name") or "unknown"
            if encoding_name.upper() != "H264":
                logger.error(
                    f"Stream {stream_id} encoding {encoding_name} not supported; expected H264"
                )
                return

            depay_sinkpad = self.elements[f'depay{stream_id}'].get_static_pad("sink")
            if not depay_sinkpad.is_linked():
                ret = pad.link(depay_sinkpad)
                if ret == Gst.PadLinkReturn.OK:
                    logger.info(f"RTSP pad linked for stream {stream_id}")

    def _configure_rtsp_source(self, source, location, stream_id):
        """Configure RTSP source with TCP transport and reconnection."""
        if not location:
            return

        source.set_property('location', location)
        source.set_property('latency', 2000)
        source.set_property('drop-on-latency', True)
        source.set_property('buffer-mode', 0)
        source.set_property('do-rtsp-keep-alive', True)

        try:
            source.set_property('protocols', GstRtsp.RTSPLowerTrans.TCP)
        except Exception as exc:
            logger.warning(f"Stream {stream_id}: failed to set TCP protocol ({exc})")
            source.set_property('protocols', 'tcp')

        tcp_timeout_us = int(self.config.get('rtsp_tcp_timeout_us', 0) or 0)
        if tcp_timeout_us > 0:
            self._set_rtsp_property(source, 'tcp-timeout', tcp_timeout_us, stream_id)

        rtsp_retry = int(self.config.get('rtsp_retry', 0) or 0)
        if rtsp_retry > 0:
            self._set_rtsp_property(source, 'retry', rtsp_retry, stream_id)

        rtsp_timeout_sec = int(self.config.get('rtsp_timeout_sec', 0) or 0)
        if rtsp_timeout_sec > 0:
            self._set_rtsp_property(source, 'timeout', rtsp_timeout_sec, stream_id)

        if self.config.get('rtsp_do_retransmission') is not None:
            self._set_rtsp_property(
                source, 'do-retransmission',
                bool(self.config.get('rtsp_do_retransmission')),
                stream_id
            )

    @staticmethod
    def _set_rtsp_property(source, name, value, stream_id):
        try:
            source.set_property(name, value)
        except Exception as exc:
            logger.warning(f"Stream {stream_id}: failed to set {name}={value}: {exc}")
