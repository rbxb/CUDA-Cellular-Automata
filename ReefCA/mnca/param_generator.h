#ifndef PARAM_GENERATOR_H
#define PARAM_GENERATOR_H

class ParamGenerator {
public:
	void generate_params(int nh_len, int pcount, unsigned int* params);
};

class ParamTuner : public ParamGenerator {
public:
	ParamTuner(int pcount, unsigned int* initial);
	void generate_params(int nh_len, int pcount, unsigned int* params);
protected:
	int pcount;
	int* initial;
};

#endif // PARAM_GENERATOR_H