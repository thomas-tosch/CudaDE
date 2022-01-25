all:
	nvcc -o programDE main.cpp DifferentialEvolutionCPU.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	rm programDE
