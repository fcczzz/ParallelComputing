#include "matrix.h"
#include <new>

void Mat::random_init(int N, int M) {
    n = N, m = M;
    a = new double[n * m];
    for (int i = 0; i < n * m; i++) {
        a[i] = 0.5 - (double)rand() / RAND_MAX;
    }
}
void Mat::zero_init(int N, int M) {
    n = N, m = M;
    a = new double[n * m];
    memset(a, 0, sizeof(double) * n * m);
}