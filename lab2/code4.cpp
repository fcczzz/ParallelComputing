#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <assert.h>

int n, *a, *b;

void Sort(int l, int r) {
    if (l == r) return;
    int mid = (l + r) >> 1;
    Sort(l, mid);
    Sort(mid + 1, r);
    int i = l, j = mid + 1, k = l;
    while (i <= mid && j <= r)
        if (a[i] <= a[j])
            b[k++] = a[i++];
        else
            b[k++] = a[j++];
    while (i <= mid) b[k++] = a[i++];
    while (j <= r) b[k++] = a[j++];
    for (int i = l; i <= r; i++) a[i] = b[i];
}

int main(int argc, char *argv[]) {
    freopen("data.in", "r", stdin);
    double t0 = omp_get_wtime();
    scanf("%d", &n);
    a = new int[n + 1];
    b = new int[n + 1];

    for (int i = 1; i <= n; i++) scanf("%d", a + i);

    Sort(1, n);
    // for (int i = 1; i <= n; i++) printf("%d\n", a[i]);

    for (int i = 1; i < n; i++) assert(a[i] <= a[i + 1]);

    double t1 = omp_get_wtime();
    printf("Time elapsed is %f.\n", t1 - t0);
    return 0;
}
