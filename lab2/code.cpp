#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char *argv[]) {
    freopen("data.in", "r", stdin);
    double t0 = omp_get_wtime();
    int n, *a;
    scanf("%d", &n);
    a = new int[n];

#pragma omp parallel for
    for (int i = 0; i < n; i++) scanf("%d", a + i);

    for (int i = 0; i < n; i++)
#pragma omp parallel for
        for (int j = i & 1; j < n - 1; j += 2)
            if (a[j] > a[j + 1]) std::swap(a[j], a[j + 1]);

    for (int i = 0; i < n - 1; i++) assert(a[i] < a[i + 1]);
    /*#pragma omp parallel private(tid)*/
    /*{*/
    /*nthreads =*/
    /*omp_get_num_threads();  // get num of threads*/
    /*tid = omp_get_thread_num(); // get my thread id*/
    /*printf("From thread %d out of %d, Hello World!\n",*/
    /*tid, nthreads);*/
    /*}*/

    double t1 = omp_get_wtime();
    printf("Time elapsed is %lf.\n", t1 - t0);
    return 0;
}
