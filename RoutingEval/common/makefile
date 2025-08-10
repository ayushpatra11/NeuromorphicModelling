# Project: Neuromorphic Routing Evaluation

# Compiler
CXX := g++
CXXFLAGS := -std=c++17 -I./ -I./nlohmann -Wall -Wextra

# Directories
BUILDDIR := ../build
SRCDIR := .
BINDIR := $(BUILDDIR)

# Sources and objects
SRC := main.cpp Utils.cpp NeuronMapper.cpp HBSNeuronMapper.cpp HBSRoutingSimulator.cpp RoutingSimulator.cpp
OBJ := $(patsubst %.cpp,$(BUILDDIR)/%.o,$(SRC))

# Target
TARGET := $(BINDIR)/RunSimulator

# Default rule
all: $(TARGET)

# Link the target
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files
$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory if not exists
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)

# Phony targets
.PHONY: all clean