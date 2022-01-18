all:
	nvcc --expt-relaxed-constexpr -o programDE main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	rm programDE
