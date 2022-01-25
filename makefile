all:
	nvcc -o programDE DifferentialEvolutionCPU.cpp main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	rm programDE
