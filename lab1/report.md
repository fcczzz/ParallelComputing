# 并行计算Lab1实验报告
方驰正PB21000163
## 一、实验目的
对OPENMP和CUDA环境的搭建。
## 二、实验过程
### 1. OPENMP环境搭建
我们使用`sudo apt-get install libomp-dev`安装OPENMP环境。
通过运行如下代码，我们可以确认OPENMP环境已经搭建成功。
```c++
int nthreads, tid;
double t0, t1;
omp_set_num_threads(4);
t0 = omp_get_wtime();
#pragma omp parallel private(tid)
{
    nthreads = omp_get_num_threads(); // get num of threads
    tid = omp_get_thread_num();       // get my thread id
    printf("From thread %d out of %d, Hello World!\n", tid, nthreads);
}
t1 = omp_get_wtime();
nthreads = omp_get_num_threads(); // get num of threads
tid = omp_get_thread_num();       // get my thread id
printf("From thread %d out of %d, Hello World!\n", tid, nthreads);

printf("Time elapsed is %f.\n", t1 - t0);
```
输出结果如下：
```shell
From thread 3 out of 4, Hello World!
From thread 2 out of 4, Hello World!
From thread 0 out of 4, Hello World!
From thread 1 out of 4, Hello World!
From thread 0 out of 1, Hello World!
Time elapsed is 0.001291.
```
说明OPENMP环境搭建成功。
### 2. CUDA环境搭建
我们使用如下指令安装CUDA环境：
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```
运行`nvcc --version`，输出
```shell
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```
可以确认CUDA环境搭建成功。
## 三、实验总结
通过本次实验，我们成功搭建了OPENMP和CUDA环境。
