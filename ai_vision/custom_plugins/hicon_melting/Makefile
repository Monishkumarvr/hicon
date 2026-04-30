################################################################################
# Makefile for HiCon Melting Detection GStreamer Plugin
################################################################################

LIB:= libgsthiconmelting.so

CXX:= g++

SRCS:= gsthiconmelting.cpp
INCS:= $(wildcard *.h)
OBJS:= $(SRCS:.cpp=.o)

CUDA_VER?= 12.6

DS_ROOT:= /opt/nvidia/deepstream/deepstream-7.1
DS_INCLUDES:= $(DS_ROOT)/sources/includes
DS_LIB:= $(DS_ROOT)/lib

LIBS_DIR:= ../../libs

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS:= -std=c++14 -Wall -O2 -fPIC
CFLAGS+= -I$(DS_INCLUDES)
CFLAGS+= -I.
CFLAGS+= -I$(LIBS_DIR)
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LIBS+= -L$(DS_LIB) \
       -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform \
       -lnvdsgst_helper -lnvds_utils -lm -lpthread \
       -L$(LIBS_DIR) -Wl,--no-as-needed -lds_white_ratio -lds_molten_detect -Wl,--as-needed \
       -Wl,-rpath,$(DS_LIB) -Wl,-rpath,'$$ORIGIN/../../libs'

INSTALL_DIR:= $(DS_LIB)/gst-plugins

.PHONY: all clean install

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

clean:
	rm -f $(OBJS) $(LIB)
