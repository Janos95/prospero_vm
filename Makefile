CXX = clang++
CXXFLAGS = -O3 -ffast-math -Xclang -fopenmp -lomp -std=c++17 -L/opt/homebrew/opt/libomp/lib
TARGET = vm
SRCS = vm.cpp

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

run-single: $(TARGET)
	OMP_NUM_THREADS=1 ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean run run-single 