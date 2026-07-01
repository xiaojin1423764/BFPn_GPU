BUILD_DIR ?= build
BUILD_TYPE ?= Release
CUDA_ARCH ?=
JOBS ?= $(shell nproc)

EXE := $(BUILD_DIR)/bin/bfp_solver

NX ?= 4000
TIME ?= 40.0
MATERIAL ?= water
ENERGY ?= 230
NY ?= 20
NZ ?= 20
NG ?= 500
NMU ?= 20
NOM ?= 20
ENERGY_MODEL ?= legacy

STREAM_DIR ?= /tmp/bfpn_stream
ENERGY_CHUNK ?= 4
LANE_CHUNK ?= 262144

CMAKE_CONFIG_ARGS := -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
ifneq ($(strip $(CUDA_ARCH)),)
CMAKE_CONFIG_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
endif

RUN_ARGS := $(NX) --time $(TIME) --material $(MATERIAL) --energy $(ENERGY) \
	--ny $(NY) --nz $(NZ) --ng $(NG) --nmu $(NMU) --nom $(NOM) \
	--energy-model $(ENERGY_MODEL)

.PHONY: all configure build run run-lite run-streaming plot analyze test clean help

all: build

configure:
	cmake $(CMAKE_CONFIG_ARGS)

build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

run: build
	$(EXE) $(RUN_ARGS)

run-lite: build
	$(EXE) $(NX) --time $(TIME) --material $(MATERIAL) --energy $(ENERGY) \
		--ny 80 --nz 80 --ng $(NG) --nmu 20 --nom 20 \
		--lite-memory

run-streaming: build
	$(EXE) $(NX) --time $(TIME) --material $(MATERIAL) --energy $(ENERGY) \
		--ny 80 --nz 80 --ng $(NG) --nmu 20 --nom 20 \
		--streaming-full --primary-only \
		--energy-chunk $(ENERGY_CHUNK) --lane-chunk $(LANE_CHUNK) \
		--stream-dir $(STREAM_DIR) --energy-model $(ENERGY_MODEL)

plot:
	python3 scripts/plot_results.py -i idd_output.txt -o idd_plot.png

analyze:
	python3 scripts/analyze_idd.py -i idd_output.txt --material $(MATERIAL) --energy $(ENERGY)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	cmake --build $(BUILD_DIR) --target clean

help:
	@echo "Targets:"
	@echo "  make              Configure and build"
	@echo "  make run          Build and run default Figure 3 development case"
	@echo "  make run-lite     Run paper-grid primary-only low-memory mode"
	@echo "  make run-streaming Run paper-grid out-of-core primary-only mode"
	@echo "  make plot         Plot idd_output.txt to idd_plot.png"
	@echo "  make analyze      Compute BP/P90/D90/D20 from idd_output.txt"
	@echo "  make test         Run CTest"
	@echo "  make clean        Clean CMake build outputs"
	@echo ""
	@echo "Common overrides:"
	@echo "  make run TIME=40 MATERIAL=water ENERGY=230 NY=20 NZ=20 NMU=20 NOM=20 ENERGY_MODEL=legacy"
	@echo "  make run CUDA_ARCH=120"
