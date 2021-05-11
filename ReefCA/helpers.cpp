/*
 * helpers.cpp
 *
 * https://github.com/rbxb/ReefCA
 */

#include "helpers.h"

#include <fstream>
#include <sstream>
#include <iomanip>

using namespace ReefCA;

template<typename T>
void ReefCA::save_pam(std::string name, T* buf, T max, int width, int height, int depth) {
    std::ofstream ofs;
    ofs.open(name, std::ios::binary);
    ofs << "P7" << std::endl;
    ofs << "WIDTH " << width << std::endl;
    ofs << "HEIGHT " << height << std::endl;
    ofs << "DEPTH " << depth << std::endl;
    ofs << "MAXVAL " << int(max) << std::endl;
    ofs << "TUPLTYPE GRAYSCALE" << std::endl;
    ofs << "ENDHDR" << std::endl;
    ofs.write((char*)buf, width * height * depth * sizeof(T));
    ofs.close();
}

std::string ReefCA::pad_image_index(int i, int n) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(n) << i;
    return ss.str();
}