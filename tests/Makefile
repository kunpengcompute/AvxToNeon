# The default platform is aarch64, use "make"
# If on x86 platform, use "make ARCH=x86"
ARCH = aarch64
CXX = g++
ARCH_CFLAGS = -march=armv8-a+fp+simd+crc

ifeq ($(ARCH), x86)
	CXX = aarch64-linux-gnu-g++
endif


ABSPATH = $(abspath $(lastword $(MAKEFILE_LIST)))
ABSDIR = $(dir $(ABSPATH))
INCLUDE_DIR = $(ABSDIR)../
DATAFILE_DIR = $(ABSDIR)../data

CXXFLAGS = -Wall -Wcast-qual -DNDEBUG -I $(INCLUDE_DIR) -I $(DATAFILE_DIR) $(ARCH_CFLAGS) -O2
LDFLAGS	= -lm

OBJS = $(wildcard $(ABSDIR)*.c)
EXEC = $(ABSDIR)test

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $^



.PHONY: clean
clean:
	$(RM) $(EXEC)

