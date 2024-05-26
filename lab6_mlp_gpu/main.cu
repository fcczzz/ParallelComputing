#include <bits/stdc++.h>
#include <ctime>
#include <vector>
#include "mnist.h"
#include "matrix.h"

struct MLP {
    Mat W1, W2, b1, b2;
    double lr;

    MLP operator=(const MLP &rhs) {
        W1 = rhs.W1;
        W2 = rhs.W2;
        b1 = rhs.b1;
        b2 = rhs.b2;
        lr = rhs.lr;
        return *this;
    }

    void init(int input_size, int hidden_size,
              int output_size, double LR) {
        W1.random_init(input_size, hidden_size, 0,
                       sqrt(2.0 / input_size));
        W2.random_init(hidden_size, output_size, 0,
                       sqrt(2.0 / hidden_size));
        b1.random_init(1, hidden_size, 0, sqrt(2.0));
        b2.random_init(1, output_size, 0, sqrt(2.0));

        lr = LR;
    }
    void forward(const Mat &input, Mat &z1, Mat &a1,
                 Mat &z2, Mat &a2) {
        // z1 = input * W1 + b1;
        // a1 = z1.relu();
        // z2 = a1 * W2 + b2;
        // a2 = z2.softmax();

        static Mat input_mult_W1(input.n, W1.m);
        Mult_mat(input, W1, input_mult_W1);
        Add_mat(input_mult_W1, b1, z1);
        Relu(z1, a1);

        static Mat a1_mult_W2(a1.n, W2.m);
        Mult_mat(a1, W2, a1_mult_W2);
        Add_mat(a1_mult_W2, b2, z2);
        Softmax(z2, a2);
    }

    void backward(Mat &input, Mat &z1, Mat &a1, Mat &z2,
                  Mat &a2, Mat &label) {
        // Mat delta2 = (a2 - label).mult(z2.softmax_());
        // Mat delta1 = (delta2 * W2.T()).mult(z1.relu_());

        // Mat dW2 = a1.T() * delta2;
        // Mat dW1 = input.T() * delta1;

        // W1 = W1 - dW1 * lr;
        // W2 = W2 - dW2 * lr;
        // b1 = b1 - delta1 * lr;
        // b2 = b2 - delta2 * lr;

        static Mat a2_sub_label(a2.n, a2.m);
        static Mat z2_softmax_(z2.n, z2.m);
        Sub_mat(a2, label, a2_sub_label);
        Softmax_(z2, z2_softmax_);
        static Mat delta2(a2_sub_label.n, z2_softmax_.m);
        Mult(a2_sub_label, z2_softmax_, delta2);

        static Mat W2_T(W2.m, W2.n);
        T(W2, W2_T);
        static Mat delta2_mult_W2_T(delta2.n, W2_T.m);
        Mult_mat(delta2, W2_T, delta2_mult_W2_T);
        static Mat z1_relu_(z1.n, z1.m);
        Relu_(z1, z1_relu_);
        static Mat delta1(delta2_mult_W2_T.n, z1_relu_.m);
        Mult(delta2_mult_W2_T, z1_relu_, delta1);

        static Mat a1_T(a1.m, a1.n);
        T(a1, a1_T);
        static Mat dW2(a1_T.n, delta2.m);
        Mult_mat(a1_T, delta2, dW2);

        static Mat input_T(input.m, input.n);
        T(input, input_T);
        static Mat dW1(input_T.n, delta1.m);
        Mult_mat(input_T, delta1, dW1);

        static Mat dW1_lr(dW1.n, dW1.m);
        Mult_num(dW1, lr, dW1_lr);
        Sub_mat(W1, dW1_lr, W1);

        static Mat dW2_lr(dW2.n, dW2.m);
        Mult_num(dW2, lr, dW2_lr);
        Sub_mat(W2, dW2_lr, W2);

        static Mat delta1_lr(delta1.n, delta1.m);
        Mult_num(delta1, lr, delta1_lr);
        Sub_mat(b1, delta1_lr, b1);

        static Mat delta2_lr(delta2.n, delta2.m);
        Mult_num(delta2, lr, delta2_lr);
        Sub_mat(b2, delta2_lr, b2);
    }

    std::pair<double, double>
    step(Mat &input, Mat &label,
         bool train = true) { // return loss and accuracy

        static Mat z1(1, W1.m), a1(1, W1.m);
        static Mat z2(1, W2.m), a2(1, W2.m);
        forward(input, z1, a1, z2, a2);

        double loss = Loss(a2, label);
        double accuracy = Accuracy(a2, label);

        if (train) backward(input, z1, a1, z2, a2, label);
        return std::make_pair(loss, accuracy);
    }
} net, ans;

void train(std::vector<mnist_data> &data, int epoch) {
    Mat *inputs = new Mat[data.size()];
    Mat *labels = new Mat[data.size()];

    for (int i = 0; i < (int)data.size(); i++) {
        static double d_input[784];
        static double d_label[10];
        for (int j = 0; j < 784; j++)
            d_input[j] = data[i].a[j] / 255.0;
        for (int j = 0; j < 10; j++) d_label[j] = 0;
        d_label[data[i].label] = 1;

        inputs[i].zero_init(1, 784);
        labels[i].zero_init(1, 10);
        cudaMemcpy(inputs[i].a, d_input,
                   784 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(labels[i].a, d_label,
                   10 * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    net.init(784, 256, 10, 0.01);
    ans = net;
    double best_accuracy = -1;
    for (int i = 0; i < epoch; i++) {
        std::vector<double> loss, accuracy;
        double st0 = 0;
        for (int j = 0; j < (int)data.size(); j++) {
            double t0 = clock();
            auto res = net.step(inputs[j], labels[j]);
            cudaDeviceSynchronize();
            double t1 = clock();
            loss.push_back(res.first);
            accuracy.push_back(res.second);

            st0 += t1 - t0;

            // if (loss.size() % 100 == 0) {
            //     printf("%d %lf %lf\n", (int)loss.size(),
            //            st0 / CLOCKS_PER_SEC,
            //            st1 / CLOCKS_PER_SEC);
            // }
        }
        double average_accuracy =
            std::accumulate(accuracy.begin(),
                            accuracy.end(), 0.0)
            / accuracy.size();
        std::cout << i << " " << average_accuracy << " "
                  << 1.0 * st0 / CLOCKS_PER_SEC
                  << std::endl;
        if (average_accuracy > best_accuracy) {
            best_accuracy = average_accuracy;
            ans = net;
        }
    }
}

std::vector<int> test(std::vector<mnist_data> &data) {
    std::cout << "test" << std::endl;
    std::vector<int> res;
    for (auto &d : data) {
        static Mat input(1, 784);

        static double *d_input = new double[784];
        for (int i = 0; i < 784; i++)
            d_input[i] = d.a[i] / 255.0;
        cudaMemcpy(input.a, d_input, 784 * sizeof(double),
                   cudaMemcpyHostToDevice);

        static Mat z1(1, ans.W1.m), a1(1, ans.W1.m);
        static Mat z2(1, ans.W2.m), a2(1, ans.W2.m);
        ans.forward(input, z1, a1, z2, a2);

        static double *a2_host = new double[10];
        cudaMemcpy(a2_host, a2.a, 10 * sizeof(double),
                   cudaMemcpyDeviceToHost);

        int predict = 0;
        for (int i = 0; i < 10; i++) {
            if (a2_host[i] > a2_host[predict]) predict = i;
        }
        res.push_back(predict);
    }
    return res;
}
int main() {
    std::cout << "start" << std::endl;
    std::vector<mnist_data> train_data =
        input("./data/train.csv", true);
    std::cout << "train_data.size()" << train_data.size()
              << std::endl;
    std::vector<mnist_data> test_data =
        input("./data/test.csv", false);
    train(train_data, 10);
    output("./output/submission.csv", test(test_data));
    return 0;
}