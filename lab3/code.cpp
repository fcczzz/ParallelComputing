#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
struct Matrix {
    int n, m;
    unsigned int **a;

    Matrix(int n = 0, int m = 0) : n(n), m(m) {
        a = new unsigned int *[n + 1];
        for (int i = 1; i <= n; i++) {
            a[i] = new unsigned int[m + 1];
            memset(a[i], 0, (m + 1) * sizeof(unsigned int));
        }
    }
    Matrix(const Matrix &_) {
        n = _.n, m = _.m;
        a = new unsigned int *[n + 1];
        for (int i = 1; i <= n; i++) {
            a[i] = new unsigned int[m + 1];
            memcpy(a[i], _.a[i], (m + 1) * sizeof(unsigned int));
        }
    }
    ~Matrix() {
        for (int i = 1; i <= n; i++) delete a[i];
        delete a;
    }
    void random_init() {
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++) a[i][j] = rand();
    }
    Matrix operator*(const Matrix &_) const {
        assert(m == _.n);
        Matrix res(n, _.m);
#pragma omp parallel for
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= _.m; j++)
                for (int k = 1; k <= m; k++) res.a[i][j] += a[i][k] * _.a[k][j];
        return res;
    }
};
const int T = 10; // do T times matrix mult
int main() {
    double t0 = omp_get_wtime();
    unsigned int sum = 0;
    // for (int i = 1; i <= T; i++) {
    // int x = rand() % 101 + 400;
    // int y = rand() % 101 + 400;
    // int z = rand() % 101 + 400;
    for (int i = 1; i <= 1; i++) {
        // int x = rand() % 101 + 400;
        // int y = rand() % 101 + 400;
        // int z = rand() % 101 + 400;
        int x = 2000, y = 2000, z = 2000;
        Matrix a(x, y), b(y, z);
        a.random_init();
        b.random_init();
        Matrix c = a * b;
        sum += c.a[1][1];
    }
    double t1 = omp_get_wtime();
    printf("sum is %u.\nTime elapsed is %lf.\n", sum, t1 - t0);
    return 0;
}
