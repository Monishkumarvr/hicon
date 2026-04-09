################################################################################
# Makefile for DeepStream Pouring Detection Application
################################################################################

APP:= deepstream-pouring-app

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++

SRCS:= deepstream_pouring_app.cpp

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.cpp=.o)

CUDA_VER?= 12.1

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes
CFLAGS+= -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= -I.
CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L/opt/nvidia/deepstream/deepstream-7.1/lib \
       -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord \
       -lnvds_utils -lnvds_msgbroker -lnvbufsurface -lnvbufsurftransform -lm \
       -lpthread \
       -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart \
       -Wl,-rpath,/opt/nvidia/deepstream/deepstream-7.1/lib

CFLAGS+= -std=c++14 -Wall -O2

.PHONY: all clean

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	@echo "Compiling: $<"
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	@echo "Linking: $@"
	$(CXX) -o $@ $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)
