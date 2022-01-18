all:
	nvcc -o programDE main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu


clean:
	rm programDE
