CXX = clang++
BATCH_UNROLL ?= 1
OPTFLAGS ?= -O3 -DNDEBUG
CXXFLAGS = $(OPTFLAGS) -Xclang -fopenmp -lomp -std=c++17 -L/opt/homebrew/opt/libomp/lib -DPROSPERO_AXIS_MODES=1 -DPROSPERO_BATCH_UNROLL=$(BATCH_UNROLL)
TARGET = vm
SRCS = main.cpp vm.cpp
EMSDK ?= $(abspath ../emsdk)
LLVM_OPENMP_SRC ?= $(abspath ../llvm-project-openmp)
OPENMP_WASM_PREFIX ?= $(abspath ../openmp-wasm-install)
OPENMP_WASM_BUILD ?= $(abspath ../openmp-wasm-build)
WASM_NODE_DIR ?= build-wasm-node
WASM_NODE_TARGET := $(WASM_NODE_DIR)/prospero.js
WASM_PROF_DIR ?= build-wasm-prof
WASM_PROF_TARGET := $(WASM_PROF_DIR)/prospero.js
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
WASM_PROF_FLAGS = $(WASM_OMP_FLAGS) -DPROSPERO_PRUNE_STATS=1 -DPROSPERO_PRUNE_TIMING=1 -DPROSPERO_PHASE_TIMING=1 --profiling-funcs --emit-symbol-map

$(TARGET): $(SRCS) vm.h simd.h Makefile
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

$(OPENMP_WASM_PREFIX)/lib/libomp.a:
	$(EMSDK_ENV) && emcmake cmake -S $(LLVM_OPENMP_SRC)/runtimes -B $(OPENMP_WASM_BUILD) -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_RUNTIMES=openmp -DLLVM_INCLUDE_TESTS=OFF -DLLVM_HOST_TRIPLE=wasm32-unknown-emscripten -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-unknown-emscripten -DCMAKE_C_FLAGS=-pthread -DCMAKE_CXX_FLAGS=-pthread -DCMAKE_ASM_FLAGS=-pthread -DOPENMP_ENABLE_OMPT_TOOLS=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF -DLIBOMP_HAVE_OMPT_SUPPORT=OFF -DLIBOMP_OMPT_SUPPORT=OFF -DLIBOMP_OMPD_SUPPORT=OFF -DLIBOMP_USE_DEBUGGER=OFF -DLIBOMP_FORTRAN_MODULES=OFF -DLIBOMP_ENABLE_SHARED=OFF -DLIBOMP_ARCH=wasm32 -DCMAKE_INSTALL_PREFIX=$(OPENMP_WASM_PREFIX) && cmake --build $(OPENMP_WASM_BUILD) --target install -j8

$(WASM_NODE_TARGET): main.cpp vm.cpp vm.h simd.h Makefile $(OPENMP_WASM_PREFIX)/lib/libomp.a
	mkdir -p $(WASM_NODE_DIR) && $(EMSDK_ENV) && em++ $(WASM_OMP_FLAGS) main.cpp vm.cpp -o $(WASM_NODE_TARGET)

wasm-omp: $(WASM_NODE_TARGET)

$(WASM_PROF_TARGET): main.cpp vm.cpp vm.h simd.h Makefile $(OPENMP_WASM_PREFIX)/lib/libomp.a
	mkdir -p $(WASM_PROF_DIR) && $(EMSDK_ENV) && em++ $(WASM_PROF_FLAGS) main.cpp vm.cpp -o $(WASM_PROF_TARGET)

wasm-prof: $(WASM_PROF_TARGET)

run: $(TARGET)
	./$(TARGET)

run-single: $(TARGET)
	OMP_NUM_THREADS=1 ./$(TARGET)

run-wasm-omp: $(WASM_NODE_TARGET)
	node $(WASM_NODE_TARGET)

run-wasm-prof: $(WASM_PROF_TARGET)
	node $(WASM_PROF_TARGET)

clean:
	rm -f $(TARGET)
	rm -rf $(WASM_NODE_DIR)
	rm -rf $(WASM_PROF_DIR)

.PHONY: clean run run-single wasm-omp wasm-prof run-wasm-omp run-wasm-prof
