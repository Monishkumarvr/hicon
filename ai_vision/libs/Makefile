# Makefile for CPU detection kernels (white-ratio + molten-blob)
# No CUDA required — plain C++ compiled with g++.

DS_ROOT ?= /opt/nvidia/deepstream/deepstream
DS_INCLUDES = $(DS_ROOT)/sources/includes

# GStreamer / GLib
GLIB_INCLUDE := $(shell pkg-config --cflags glib-2.0 2>/dev/null || echo "-I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include")
GST_INCLUDES = -I/usr/include/gstreamer-1.0 $(GLIB_INCLUDE)

CXX = g++
CXXFLAGS = -std=c++14 -fPIC -Wall -Wno-unused-function -O2
INCLUDES = -I$(DS_INCLUDES) $(GST_INCLUDES)

WHITE_RATIO_TARGET = libds_white_ratio.so
WHITE_RATIO_OBJECTS = white_ratio_bridge.o white_ratio_cuda_kernel.o

MOLTEN_TARGET = libds_molten_detect.so
MOLTEN_OBJECTS = molten_detect_bridge.o molten_detect_cuda_kernel.o

all: $(WHITE_RATIO_TARGET) $(MOLTEN_TARGET)

white_ratio_bridge.o: white_ratio_bridge.cpp white_ratio_cuda.h white_ratio_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

white_ratio_cuda_kernel.o: white_ratio_cuda_kernel.cu white_ratio_cuda.h white_ratio_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -x c++ -c $< -o $@

$(WHITE_RATIO_TARGET): $(WHITE_RATIO_OBJECTS)
	$(CXX) -shared -o $@ $^ -lgstreamer-1.0 -lglib-2.0
	@echo "Built $(WHITE_RATIO_TARGET)"

molten_detect_bridge.o: molten_detect_bridge.cpp molten_detect.h molten_detect_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

molten_detect_cuda_kernel.o: molten_detect_cuda_kernel.cu molten_detect.h molten_detect_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -x c++ -c $< -o $@

$(MOLTEN_TARGET): $(MOLTEN_OBJECTS)
	$(CXX) -shared -o $@ $^ -lgstreamer-1.0 -lglib-2.0
	@echo "Built $(MOLTEN_TARGET)"

clean:
	rm -f $(WHITE_RATIO_TARGET) $(WHITE_RATIO_OBJECTS) $(MOLTEN_TARGET) $(MOLTEN_OBJECTS)
	@echo "Cleaned"

.PHONY: all clean
