# SPDX-License-Identifier: BSD-3-Clause

UNAME_S := $(shell uname -s)

ifeq ($(origin CC), default)
  ifeq ($(UNAME_S),Darwin)
    CC := clang
  else
    CC := cc
  endif
endif

CFLAGS := -O3 -std=c11 -fPIC -fvisibility=hidden \
          -Wall -Wextra -Wpedantic \
          -Wshadow -Wconversion \
          -Wno-unused-parameter \
          -MMD -MP\
	  -Wno-c23-extensions

ifeq ($(UNAME_S),Darwin)
  CFLAGS += -mcpu=native
  BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null)
  INCLUDES := -I$(BREW_PREFIX)/include -I$(BREW_PREFIX)/opt/libomp/include -Iinclude
  LDFLAGS := -L$(BREW_PREFIX)/lib -L$(BREW_PREFIX)/opt/libomp/lib \
             -Wl,-rpath,$(BREW_PREFIX)/lib -Wl,-rpath,$(BREW_PREFIX)/opt/libomp/lib
  LIBS := -lgsl -lgslcblas -lcerf -lomp -lm
  OMPFLAGS := -Xpreprocessor -fopenmp
  TARGET := libsalt.dylib
else
  CFLAGS += -march=native
  INCLUDES := -Iinclude
  LDFLAGS := -fopenmp
  LIBS := -lgsl -lgslcblas -lcerf -lm
  OMPFLAGS := -fopenmp
  TARGET := libsalt.so
endif

SRC := src/SALT2026_LineProfile.c src/SALT2026_Emission.c src/SALT2026_Absorption.c \
       src/SALT_Inflow_LineProfile.c src/SALT_Inflow_Emission.c \
       src/SALT_Inflow_Absorption.c
OBJ := $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) -shared -o $@ $^ $(LDFLAGS) $(LIBS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) $(OMPFLAGS) -c $< -o $@

debug: CFLAGS := $(filter-out -O3,$(CFLAGS)) -O0 -g
debug: clean $(TARGET)

clean:
	rm -f $(OBJ) $(OBJ:.o=.d) $(TARGET)

-include $(OBJ:.o=.d)

.PHONY: all clean debug
