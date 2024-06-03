// its a mnist reader
#include "mnist.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::vector<mnist_data> input(std::string filename,
                              bool is_train) {
    std::ifstream file(filename, std::ios::binary);
    std::string line;
    std::vector<mnist_data> data;
    getline(file, line);
    while (getline(file, line)) {
        //读入数据，以,分割
        std::istringstream stream(line);
        std::string s;
        mnist_data tmp;
        if (is_train) {
            getline(stream, s, ',');
            tmp.label = (unsigned char)std::stoi(s);
        }
        int index = 0;
        while (getline(stream, s, ',')) {
            tmp.a[index++] = (unsigned char)std::stoi(s);
        }
        data.push_back(tmp);
    }
    return data;
}

void output(std::string filename, std::vector<int> res) {
    std::ofstream file(filename);
    file << "ImageId,Label\n";
    for (int i = 0; i < (int)res.size(); i++) {
        file << i + 1 << "," << res[i] << "\n";
    }
    file.close();
}