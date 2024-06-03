#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <assert.h>
#include <cstring>

int n, *a, *b;

void Merge(int l1, int r1, int l2, int r2) {
    int i = l1, j = l2, k = l1;
    while (i <= r1 && j <= r2)
        if (a[i] <= a[j])
            b[k++] = a[i++];
        else
            b[k++] = a[j++];
    while (i <= r1) b[k++] = a[i++];
    while (j <= r2) b[k++] = a[j++];
    for (int i = l1; i <= r2; i++) a[i] = b[i];
}

void Sort(int l, int r) {
#pragma omp parallel for
    for (int i = l; i < r; i += 2)
        if (a[i] > a[i + 1]) std::swap(a[i], a[i + 1]);

    for (int len = 2; len <= (r - l + 1); len <<= 1) {
#pragma omp parallel for
        for (int i = l; i <= r - len; i += 2 * len)
            Merge(i, i + len - 1, i + len,
                  std::min(i + 2 * len - 1, r));
    }
}

int main(int argc, char *argv[]) {
    freopen("data.in", "r", stdin);
    double t0 = omp_get_wtime();
    scanf("%d", &n);
    a = new int[n + 1];
    b = new int[n + 1];

    for (int i = 1; i <= n; i++) scanf("%d", a + i);

    Sort(1, n);

    for (int i = 1; i < n; i++) assert(a[i] <= a[i + 1]);

    double t1 = omp_get_wtime();
    printf("Time elapsed is %f.\n", t1 - t0);
    return 0;
}
