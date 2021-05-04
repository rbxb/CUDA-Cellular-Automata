#include "generate_mnca.cu"

#define FRAMES 200
#define SIMS 1000

class saveFrame : public SaveFrame {
public:
	bool save_frame(int i) {
		if (i == FRAMES - 1) return true;
		return false;
	}
};

int main(void) {
	//unsigned int initial[NUM_PARAMS] = { 7,12,19,21,10,25,53,133 };
	//ParamGenerator* paramg = new ParamTuner(NUM_PARAMS, initial);
	ParamGenerator* paramg = new ParamGenerator();
	generate_mnca(paramg, SIMS, FRAMES, new saveFrame(), "out");
}