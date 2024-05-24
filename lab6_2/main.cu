#include <bits/stdc++.h>
#include "mnist.h"
#include "matrix.h"

struct MLP {
    Mat W1, W2, b1, b2;
    int lr;
    void init(int input_size, int hidden_size,
              int output_size, int LR) {
        W1.random_init(hidden_size, input_size);
        W2.random_init(output_size, hidden_size);
        b1.random_init(hidden_size, 1);
        b2.random_init(output_size, 1);

        lr = LR;
    }
    void forward(Mat &input, Mat &z1, Mat &a1, Mat &z2,
                 Mat &a2) {
        z1 = W1 * input + b1;
        a1 = z1.relu();
        z2 = W2 * a1 + b2;
        a2 = z2.softmax();
    }

    void backward(Mat &input, Mat &z1, Mat &a1, Mat &z2,
                  Mat &a2, Mat &label) {
        Mat delta1 = (a2 - label) * z2.softmax_();
        Mat delta2 = W2.T() * delta1 * z1.relu_();

        Mat dW2 = delta1 * a1.T();
        Mat dW1 = delta2 * input.T();
        Mat db2 = delta1.sum();
        Mat db1 = delta2.sum();

        W1 = W1 - dW1 * lr;
        W2 = W2 - dW2 * lr;
        b1 = b1 - db1 * lr;
        b2 = b2 - db2 * lr;
    }


    void step(Mat &input, Mat &label) {
        Mat z1, a1, z2, a2;
        forward(input, z1, a1, z2, a2);
        backward(input, z1, a1, z2, a2, label);
    }
    
    void train(std::vector<mnist_data> &data, int epoch) {
        for (int i = 0; i < epoch; i++) {}
    }
};

int main() {
    std::vector<mnist_data> train_data =
        input("./data/train.csv", true);
    return 0;
}