################################################################################
# Makefile for HiCon Pouring Detection GStreamer Plugin
#
# Builds a shared library (libgsthiconpouring.so) that can be loaded
# by GStreamer as a plugin element: hicon_pouring_detect
#
# Build:   make
# Install: make install
# Clean:   make clean
# Test:    GST_PLUGIN_PATH=. gst-inspect-1.0 hicon_pouring_detect
################################################################################

LIB:= libgsthiconpouring.so
TEST_BIN:= test_association_logic

CXX:= g++

SRCS:= gsthiconpouring.cpp association_logic.cpp
INCS:= $(wildcard *.h)
OBJS:= $(SRCS:.cpp=.o)

CUDA_VER?= 12.6

DS_ROOT:= /opt/nvidia/deepstream/deepstream-7.1
DS_INCLUDES:= $(DS_ROOT)/sources/includes
DS_LIB:= $(DS_ROOT)/lib

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS:= -std=c++14 -Wall -O2 -fPIC
CFLAGS+= -I$(DS_INCLUDES)
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= -I.
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LIBS+= -L$(DS_LIB) \
       -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
       -lnvdsgst_helper -lnvds_utils -lm -lpthread \
       -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart \
       -Wl,-rpath,$(DS_LIB)

# Default install location — DeepStream's plugin directory
INSTALL_DIR:= $(DS_LIB)/gst-plugins

.PHONY: all clean install test-association

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo "  CXX   $<"
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) Makefile
	@echo "  LINK  $@"
	$(CXX) -shared -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	@echo "  INSTALL $(INSTALL_DIR)/$(LIB)"
	cp -f $(LIB) $(INSTALL_DIR)/

$(TEST_BIN): association_logic.cpp association_logic.h test_association_logic.cpp
	@echo "  CXX   $@"
	$(CXX) -std=c++14 -Wall -O2 -I. -o $@ association_logic.cpp test_association_logic.cpp

test-association: $(TEST_BIN)
	@echo "  RUN   $(TEST_BIN)"
	./$(TEST_BIN)

clean:
	rm -f $(OBJS) $(LIB) $(TEST_BIN)
