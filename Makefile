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
CONV_NXS ?= 500,1000,2000,4000
CONV_OUT ?= results/convergence_strang
CONV_PRIMARY_ONLY ?= 0
CONV_ENERGY_ONLY ?= 0
CONV_NO_TRANSPORT ?= 0
CONV_NO_ANGLE ?= 0
CONV_NO_SPATIAL_CLIPPING ?= 0
CONV_HEARTBEAT ?= 30

CONV_PRIMARY_ONLY_ARG :=
ifeq ($(CONV_PRIMARY_ONLY),1)
CONV_PRIMARY_ONLY_ARG := --primary-only
endif
CONV_ENERGY_ONLY_ARG :=
ifeq ($(CONV_ENERGY_ONLY),1)
CONV_ENERGY_ONLY_ARG := --energy-only
endif
CONV_NO_TRANSPORT_ARG :=
ifeq ($(CONV_NO_TRANSPORT),1)
CONV_NO_TRANSPORT_ARG := --no-transport
endif
CONV_NO_ANGLE_ARG :=
ifeq ($(CONV_NO_ANGLE),1)
CONV_NO_ANGLE_ARG := --no-angle
endif
CONV_NO_SPATIAL_CLIPPING_ARG :=
ifeq ($(CONV_NO_SPATIAL_CLIPPING),1)
CONV_NO_SPATIAL_CLIPPING_ARG := --no-spatial-clipping
endif

CMAKE_CONFIG_ARGS := -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
ifneq ($(strip $(CUDA_ARCH)),)
CMAKE_CONFIG_ARGS += -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
endif

RUN_ARGS := $(NX) --time $(TIME) --material $(MATERIAL) --energy $(ENERGY) \
	--ny $(NY) --nz $(NZ) --ng $(NG) --nmu $(NMU) --nom $(NOM) \
	--energy-model $(ENERGY_MODEL)

.PHONY: all configure build run run-lite run-streaming convergence paper-convergence plot analyze test clean help

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

convergence: build
	python3 scripts/convergence_order.py --exe $(EXE) --nxs $(CONV_NXS) \
		--out $(CONV_OUT) --time $(TIME) --material $(MATERIAL) \
		--energy $(ENERGY) --ny $(NY) --nz $(NZ) --ng $(NG) \
		--nmu $(NMU) --nom $(NOM) --energy-model $(ENERGY_MODEL) \
		--heartbeat $(CONV_HEARTBEAT) $(CONV_PRIMARY_ONLY_ARG) \
		$(CONV_ENERGY_ONLY_ARG) $(CONV_NO_TRANSPORT_ARG) \
		$(CONV_NO_ANGLE_ARG) $(CONV_NO_SPATIAL_CLIPPING_ARG)

paper-convergence: build
	python3 scripts/convergence_order.py --exe $(EXE) --nxs $(CONV_NXS) \
		--out $(CONV_OUT) --time $(TIME) --material $(MATERIAL) \
		--energy $(ENERGY) --ny $(NY) --nz $(NZ) --ng $(NG) \
		--nmu $(NMU) --nom $(NOM) --energy-model $(ENERGY_MODEL) \
		--heartbeat $(CONV_HEARTBEAT) --save-energy-moments \
		$(CONV_PRIMARY_ONLY_ARG) $(CONV_ENERGY_ONLY_ARG) \
		$(CONV_NO_TRANSPORT_ARG) $(CONV_NO_ANGLE_ARG) \
		$(CONV_NO_SPATIAL_CLIPPING_ARG)
	python3 scripts/paper_convergence_order.py --out $(CONV_OUT) \
		--nxs $(CONV_NXS) --ng $(NG)

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
	@echo "  make convergence  Run Nx refinement, compute errors, and plot observed order"
	@echo "  make paper-convergence Run paper Eq.25 psi1/psi2 convergence diagnostics"
	@echo "  make plot         Plot idd_output.txt to idd_plot.png"
	@echo "  make analyze      Compute BP/P90/D90/D20 from idd_output.txt"
	@echo "  make test         Run CTest"
	@echo "  make clean        Clean CMake build outputs"
	@echo ""
	@echo "Common overrides:"
	@echo "  make run TIME=40 MATERIAL=water ENERGY=230 NY=20 NZ=20 NMU=20 NOM=20 ENERGY_MODEL=legacy"
	@echo "  make convergence CONV_NXS=250,500,1000,2000 ENERGY_MODEL=eq15"
	@echo "  make paper-convergence CONV_OUT=results/paper_conv CONV_NXS=1600,3200,6400"
	@echo "  make convergence CONV_PRIMARY_ONLY=1"
	@echo "  make paper-convergence CONV_ENERGY_ONLY=1"
	@echo "  make paper-convergence CONV_NO_TRANSPORT=1"
	@echo "  make paper-convergence CONV_NO_ANGLE=1"
	@echo "  make paper-convergence CONV_NO_SPATIAL_CLIPPING=1"
	@echo "  make convergence CONV_HEARTBEAT=10"
	@echo "  make run CUDA_ARCH=120"
