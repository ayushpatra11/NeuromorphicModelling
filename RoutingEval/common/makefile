# Project: Neuromorphic Routing Evaluation

# Compiler
CXX := g++
CXXFLAGS := -std=c++17 -I./ -I./nlohmann -Wall -Wextra

# Sources and objects
SRC := main.cpp Utils.cpp NeuronMapper.cpp RoutingSimulator.cpp
OBJ := $(SRC:.cpp=.o)

# Target
TARGET := RunSimulator

# Default rule
all: $(TARGET)

# Link the target
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)

# Phony targets
.PHONY: all clean
