#include "param_generator.h"

void ParamGenerator::generate_params(int nh_len, int pcount, unsigned int* params) {
	unsigned int v = 0;
	for (int i = 0; i < pcount; i++) {
		unsigned int range = nh_len - v;
		v = rand() % (range * 2 / 3) + v;
		params[i] = v;
	}
}

ParamTuner::ParamTuner(int pcount, unsigned int* initial) : ParamGenerator() {
	this->pcount = pcount;
	this->initial = new int[pcount];
	for (int i = 0; i < pcount; i++) {
		this->initial[i] = initial[i];
	}
}

void ParamTuner::generate_params(int nh_len, int pcount, unsigned int* params) {
	for (int i = 0; i < pcount; i++) {
		int initial_p = initial[i];
		int max_shift = (int(initial[i] * 0.20) + 1) * 2;
		int new_p = initial_p + rand() % max_shift - max_shift / 2;
		if (new_p < 0) new_p = 0;
		params[i] = new_p;
	}
}
