all:
	nvcc -o programDE main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu --expt-relaxed-constexpr


clean:
	rm programDE
