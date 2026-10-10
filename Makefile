# Taffy time stretch / pitch shift engine. Direct build; CMakeLists.txt also works.
CXX      ?= c++
CC       ?= cc
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Iinclude
CFLAGS   ?= -O2 -std=c11 -Wall -Iinclude
LDFLAGS  ?=

BUILD := build

LIB_SRC := src/fft.cpp src/engine.cpp
LIB_OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(LIB_SRC))

.PHONY: all lib clean test

all: lib taffy_stretch

lib: $(LIB_OBJ)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -- Public API example CLI (examples/taffy_stretch.c) -----------------------
# The example is C, so compile it separately, otherwise c++ treats .c as C++
# and emits a deprecation warning.
taffy_stretch: $(BUILD)/taffy_stretch.o $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD)/taffy_stretch.o: examples/taffy_stretch.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

test: taffy_stretch
	./taffy_stretch

clean:
	rm -rf $(BUILD) taffy_stretch
