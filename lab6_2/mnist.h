#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
struct mnist_data {
    unsigned char label;
    unsigned char a[28 * 28];
};
std::vector<mnist_data> input(std::string filename,
                              bool is_train);
void output(std::string filename, std::vector<int> data);