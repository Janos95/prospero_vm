BATCH_UNROLL ?= 1
OPTFLAGS ?= -O3 -DNDEBUG
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
# Apple clang with Homebrew's libomp (brew install libomp)
CXX = clang++
OPENMP_FLAGS ?= -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
else
# GCC or clang with an OpenMP runtime installed
OPENMP_FLAGS ?= -fopenmp
endif
CXXFLAGS = $(OPTFLAGS) $(OPENMP_FLAGS) -std=c++17 -DPROSPERO_AXIS_MODES=1 -DPROSPERO_BATCH_UNROLL=$(BATCH_UNROLL)
TARGET = vm
SRCS = main.cpp vm.cpp
FRONTIER_TARGET = frontier
FRONTIER_SRCS = frontier_main.cpp frontier.cpp
# The term-based renderer reproduces the published image exactly only without FMA contraction.
FRONTIER_CXXFLAGS = $(OPTFLAGS) -std=c++17 -ffp-contract=off
EMSDK ?= $(abspath ../emsdk)
LLVM_OPENMP_SRC ?= $(abspath ../llvm-project-openmp)
OPENMP_WASM_PREFIX ?= $(abspath ../openmp-wasm-install)
OPENMP_WASM_BUILD ?= $(abspath ../openmp-wasm-build)
WASM_NODE_DIR ?= build-wasm-node
WASM_NODE_TARGET := $(WASM_NODE_DIR)/prospero.js
WASM_PROF_DIR ?= build-wasm-prof
WASM_PROF_TARGET := $(WASM_PROF_DIR)/prospero.js
WASM_WEB_DIR ?= build-wasm-web
WASM_WEB_TARGET := $(WASM_WEB_DIR)/prospero.js
EMSDK_ENV = export EMSDK_QUIET=1; . $(EMSDK)/emsdk_env.sh >/dev/null
WASM_OMP_FLAGS = $(OPTFLAGS) -std=c++17 -DPROSPERO_AXIS_MODES=1 -DPROSPERO_BATCH_UNROLL=$(BATCH_UNROLL) -fopenmp -pthread \
	-msimd128 \
	-I$(OPENMP_WASM_PREFIX)/include \
	$(OPENMP_WASM_PREFIX)/lib/libomp.a \
	-sPTHREAD_POOL_SIZE=8 \
	-sPROXY_TO_PTHREAD \
	-sNODERAWFS=1 \
	-sINITIAL_MEMORY=268435456 \
	-sEXIT_RUNTIME=1
# Browser build of the term-based renderer, matching the blog post's runner.js
# (window.createProsperoModule, wasm embedded as base64 in prospero-binary.js).
WASM_WEB_FLAGS = $(OPTFLAGS) -std=c++17 -ffp-contract=off -msimd128 \
	-sMODULARIZE=1 -sEXPORT_NAME=createProsperoModule -sENVIRONMENT=web \
	-sEXPORTED_FUNCTIONS=_prospero_load,_prospero_set_threads,_prospero_benchmark,_prospero_std_ms,_prospero_min_ms,_prospero_max_ms,_prospero_instructions,_prospero_frontier_terms,_prospero_checksum,_prospero_image_data,_malloc,_free \
	-sEXPORTED_RUNTIME_METHODS=ccall,cwrap,stringToUTF8,lengthBytesUTF8,HEAPU8
WASM_PROF_FLAGS = $(WASM_OMP_FLAGS) -DPROSPERO_PRUNE_STATS=1 -DPROSPERO_PRUNE_TIMING=1 -DPROSPERO_PHASE_TIMING=1 --profiling-funcs --emit-symbol-map

$(TARGET): $(SRCS) vm.h simd.h Makefile
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

$(FRONTIER_TARGET): $(FRONTIER_SRCS) frontier.h Makefile
	$(CXX) $(FRONTIER_CXXFLAGS) -o $(FRONTIER_TARGET) $(FRONTIER_SRCS)

$(OPENMP_WASM_PREFIX)/lib/libomp.a:
	$(EMSDK_ENV) && emcmake cmake -S $(LLVM_OPENMP_SRC)/runtimes -B $(OPENMP_WASM_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_RUNTIMES=openmp -DLLVM_INCLUDE_TESTS=OFF -DLLVM_HOST_TRIPLE=wasm32-unknown-emscripten -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-unknown-emscripten -DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread -DCMAKE_ASM_FLAGS=-pthread -DOPENMP_ENABLE_OMPT_TOOLS=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DLIBOMP_HAVE_OMPT_SUPPORT=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_OMPD_SUPPORT=OFF -DLIBOMP_USE_DEBUGGER=OFF -DLIBOMP_FORTRAN_MODULES=OFF -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_ARCH=wasm32 -DCMAKE_INSTALL_PREFIX=$(OPENMP_WASM_PREFIX) && cmake --build $(OPENMP_WASM_BUILD) --target install -j8

$(WASM_NODE_TARGET): main.cpp vm.cpp vm.h simd.h Makefile $(OPENMP_WASM_PREFIX)/lib/libomp.a
	mkdir -p $(WASM_NODE_DIR) && $(EMSDK_ENV) && em++ $(WASM_OMP_FLAGS) main.cpp vm.cpp -o $(WASM_NODE_TARGET)

wasm-omp: $(WASM_NODE_TARGET)

$(WASM_PROF_TARGET): main.cpp vm.cpp vm.h simd.h Makefile $(OPENMP_WASM_PREFIX)/lib/libomp.a
	mkdir -p $(WASM_PROF_DIR) && $(EMSDK_ENV) && em++ $(WASM_PROF_FLAGS) main.cpp vm.cpp -o $(WASM_PROF_TARGET)

wasm-prof: $(WASM_PROF_TARGET)

$(WASM_WEB_TARGET): frontier.cpp frontier.h web.cpp Makefile
	mkdir -p $(WASM_WEB_DIR) && $(EMSDK_ENV) && em++ $(WASM_WEB_FLAGS) frontier.cpp web.cpp -o $(WASM_WEB_TARGET)
	printf 'window.PROSPERO_WASM_BASE64 = "%s";\n' "$$(base64 < $(WASM_WEB_DIR)/prospero.wasm | tr -d '\n')" > $(WASM_WEB_DIR)/prospero-binary.js

wasm-web: $(WASM_WEB_TARGET)

run: $(TARGET)
	./$(TARGET)

run-single: $(TARGET)
	OMP_NUM_THREADS=1 ./$(TARGET)

run-frontier: $(FRONTIER_TARGET)
	./$(FRONTIER_TARGET)

# Checks that the term-based renderer still reproduces the published image.
check: $(FRONTIER_TARGET)
	./$(FRONTIER_TARGET) > /dev/null && echo "frontier: checksum and term count match"

run-wasm-omp: $(WASM_NODE_TARGET)
	node $(WASM_NODE_TARGET)

run-wasm-prof: $(WASM_PROF_TARGET)
	node $(WASM_PROF_TARGET)

clean:
	rm -f $(TARGET) $(FRONTIER_TARGET)
	rm -rf $(WASM_WEB_DIR)
	rm -rf $(WASM_NODE_DIR)
	rm -rf $(WASM_PROF_DIR)

.PHONY: clean check run run-single run-frontier wasm-omp wasm-prof wasm-web run-wasm-omp run-wasm-prof
