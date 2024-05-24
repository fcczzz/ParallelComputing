// its a mnist reader
#include "mnist.h"
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
    int index = 0;
    while (getline(file, line)) {
        std::vector<int> v;
        //读入数据，以,分割
        std::istringstream stream(line);
        std::string s;
        while (getline(stream, s, ',')) {
            v.push_back(stoi(s));
        }

        if (data.empty()) {
            data = std::vector<mnist_data>(v.size());
        }
        if (is_train) {
            for (int i = 0; i < (int)v.size(); i++)
                data[i].label = v[i];
            is_train = false;
        } else {
            for (int i = 0; i < (int)v.size(); i++)
                data[i].a[index] = v[i];
            ++index;
        }
    }
    return data;
}