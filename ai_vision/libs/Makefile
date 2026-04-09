# Makefile for CUDA detection kernels (white-ratio + molten-blob)
# Build on Jetson Orin Nano (aarch64, JetPack 6.2.1, CUDA 12.6)

DS_ROOT ?= /opt/nvidia/deepstream/deepstream
DS_INCLUDES = $(DS_ROOT)/sources/includes

CUDA_ROOT ?= /usr/local/cuda
CUDA_INCLUDES = $(CUDA_ROOT)/include
CUDA_LIB = $(CUDA_ROOT)/lib64

# GStreamer / GLib — detect include paths for aarch64 vs x86_64
GLIB_INCLUDE := $(shell pkg-config --cflags glib-2.0 2>/dev/null || echo "-I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include")
GST_INCLUDES = -I/usr/include/gstreamer-1.0 $(GLIB_INCLUDE)

CXX = g++
NVCC = $(CUDA_ROOT)/bin/nvcc
CXXFLAGS = -std=c++14 -fPIC -Wall -Wno-unused-function -O2
NVCCFLAGS = -std=c++14 -O2 -Xcompiler -fPIC
INCLUDES = -I$(DS_INCLUDES) -I$(CUDA_INCLUDES) $(GST_INCLUDES)

WHITE_RATIO_TARGET = libds_white_ratio.so
WHITE_RATIO_OBJECTS = white_ratio_bridge.o white_ratio_cuda_kernel.o

MOLTEN_TARGET = libds_molten_detect.so
MOLTEN_OBJECTS = molten_detect_bridge.o molten_detect_cuda_kernel.o

all: $(WHITE_RATIO_TARGET) $(MOLTEN_TARGET)

white_ratio_bridge.o: white_ratio_bridge.cpp white_ratio_cuda.h white_ratio_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

white_ratio_cuda_kernel.o: white_ratio_cuda_kernel.cu white_ratio_cuda.h white_ratio_cuda_kernel.h
	$(NVCC) $(NVCCFLAGS) -I$(DS_INCLUDES) -I$(CUDA_INCLUDES) -c $< -o $@

$(WHITE_RATIO_TARGET): $(WHITE_RATIO_OBJECTS)
	$(CXX) -shared -o $@ $^ -L$(CUDA_LIB) -lcudart -lgstreamer-1.0 -lglib-2.0
	@echo "Built $(WHITE_RATIO_TARGET)"

molten_detect_bridge.o: molten_detect_bridge.cpp molten_detect.h molten_detect_cuda_kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

molten_detect_cuda_kernel.o: molten_detect_cuda_kernel.cu molten_detect.h molten_detect_cuda_kernel.h
	$(NVCC) $(NVCCFLAGS) -I$(DS_INCLUDES) -I$(CUDA_INCLUDES) -c $< -o $@

$(MOLTEN_TARGET): $(MOLTEN_OBJECTS)
	$(CXX) -shared -o $@ $^ -L$(CUDA_LIB) -lcudart -lgstreamer-1.0 -lglib-2.0
	@echo "Built $(MOLTEN_TARGET)"

clean:
	rm -f $(WHITE_RATIO_TARGET) $(WHITE_RATIO_OBJECTS) $(MOLTEN_TARGET) $(MOLTEN_OBJECTS)
	@echo "Cleaned"

.PHONY: all clean
