all:
	nvcc -o programDE main.cpp DifferentialEvolutionGPU.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	rm programDE
