#include "mnca.cu"

std::vector<int> generate_nh(int max_r, int max_rings) {
	std::vector<int> nh = std::vector<int>();
	int rings = rand() % max_rings + 1;
	for (int i = 0; i < rings; i++) {
		int r = rand() % max_r;
		mnca::generate_nh_midpoint_circle(r, nh);
	}
	return nh;
}
