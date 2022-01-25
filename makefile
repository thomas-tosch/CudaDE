all:
	nvcc -o programDE main.cpp DifferentialEvolution.cpp DifferentialEvolutionGPU.cu DifferentialEvolutionGPU.cpp


clean:
	rm programDE
