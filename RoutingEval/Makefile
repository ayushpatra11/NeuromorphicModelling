# ============================================================
# RoutingEval – Makefile
# ============================================================
# Delegates to the canonical makefile in common/ which keeps
# the source files and build logic co-located.
#
# Usage:
#   make             build the simulator
#   make clean       remove build artefacts
#   make run         build + run the simulator

.PHONY: all clean run

all:
	$(MAKE) -C common

clean:
	$(MAKE) -C common clean

run: all
	./build/RunSimulator
