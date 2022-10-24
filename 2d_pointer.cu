/**
 * @file 2d_pointer.cu
 * @author upupwords@outlook.com
 * @brief 测试二级指针的访问和传输
 * @version 0.1
 * @date 2022-10-24
 * 
 * @copyright Copyright (c) 2022
 * 
 */



#include <cuda_runtime.h>
#include <cublas_v2.h>     
#include <iostream>
#include <stdio.h>

using namespace std;


__global__
void show(float** ptr, int size)
{
    printf("hh\n");
    printf("%f\n", ptr[0][0]);
    printf("%f\n", ptr[0][1]);
    printf("%f\n", ptr[1][0]);
    printf("%f\n", ptr[1][1]);
        // for(int i =0; i<size; i++)
        // printf("%f\n", ptr[i]);
}
__global__
void show2(float* ptr, int size)
{
    printf("hhhhhh\n");
    printf("%f\n", ptr[0]);
    printf("%f\n", ptr[1]);
    
        // for(int i =0; i<size; i++)
        // printf("%f\n", ptr[i]);
}


int main()
{
    float ** a;
    
    cudaMallocManaged(&a, sizeof(float*)*2, cudaMemAttachGlobal);
    cudaMalloc((void **)&a[0], sizeof(float)*2);
    cudaMalloc((void **)&a[1], sizeof(float)*2);
    float a_h[2][2]={1,2,3,4};
    cudaMemcpy(a[0],a_h[0],sizeof(float)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(a[1],a_h[1],sizeof(float)*2, cudaMemcpyHostToDevice);
    show<<<1,1>>>(a, 2);
    show2<<<1,1>>>(a[0], 2);
    show2<<<1,1>>>(a[1], 2);    
    cudaDeviceSynchronize();

    // float ** a= new float* [2];
    // // cudaMalloc((void ***)&a, sizeof(float*)*2);
    // // cudaMallocManaged(&a, sizeof(float*)*2, cudaMemAttachGlobal);
    // cudaMalloc((void **)&a[0], sizeof(float)*2);
    // cudaMalloc((void **)&a[1], sizeof(float)*2);
    //  float a_h[2][2]={1,2,3,4};
   
    
    // cudaMemcpy(a[0],a_h[0],sizeof(float)*2, cudaMemcpyHostToDevice);
    // cudaMemcpy(a[1],a_h[1],sizeof(float)*2, cudaMemcpyHostToDevice);

    // float **host;
    // cudaMalloc((void ***)&host, sizeof(float*)*2);
    // cudaMemcpy(host,a,sizeof(float*)*2, cudaMemcpyHostToDevice);

    // show<<<1,1>>>(host[1], 2);
        
    // cudaDeviceSynchronize();
   

   return 0;
}




// int main()
// {
//    float a[12]={1,7,2,8,3,9,4,10,5,11,6,12};
//    float b[24]={ 2,6,10,
//                  3,7,11,
//                  4,8,12,
//                  5,9,13,
//                  1,5,9,
//                  2,6,10,
//                  3,7,11,
//                  4,8,12};
//    float c[16]={100,200,300,400,500,700};
//    // float a[8]={1,5,2,6,3,7,4,8};
//    // float b[8]={2,6,3,7,4,8,5,9};
//    // float c[8];
   
//    float* d_a, *d_b, *d_c;
   
//    cudaMalloc(&d_a, sizeof(float)*12);
//    cudaMalloc(&d_b, sizeof(float)*24);
//    cudaMalloc(&d_c, sizeof(float)*16);

//    cudaMemcpy(d_a, a, sizeof(float)*12, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, b, sizeof(float)*24, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_c, c, sizeof(float)*16, cudaMemcpyHostToDevice);


//    cublasHandle_t handle;
//    cublasStatus_t ret;
//    ret = cublasCreate(&handle);
//    float *a_array[2], *b_array[2];
//    float *c_array[2];
//    a_array[0]=d_a;
//    a_array[1]=d_a+4;
//    b_array[0]=d_b;
//    b_array[1]=d_b+4;
//    c_array[0]=d_c;
//    c_array[1]=d_c+4;
   
//    float **d_Marray, **d_Narray;
//    float **d_Parray;
//    cudaMalloc((void**)&d_Marray, 8*sizeof(float *));
//    cudaMalloc((void**)&d_Narray, 8*sizeof(float *));
//    cudaMalloc((void**)&d_Parray, 8*sizeof(float *));
//    cudaMemcpy(d_Marray, a_array, 8*sizeof(float *), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_Narray, b_array, 8*sizeof(float *), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_Parray, c_array, 8*sizeof(float *), cudaMemcpyHostToDevice);


//    float alpha  =  1.0f;
//    float beta  =  0.0f;
//    int m = 2;
//    int n = 4;
//    int k = 3;
//    int lda = 2;
//    int ldb = 3;
//    int ldc = 2;
//    int batch = 2;
//    ret = cublasSgemmStridedBatched(handle,
//                      CUBLAS_OP_N,
//                      CUBLAS_OP_N,
//                      m,n,k,
//                      &alpha,
//                      d_a,  lda,
//                      6,
//                      d_b,  ldb,
//                      12,
//                      &beta,
//                      d_c,  ldc,
//                      8,
//                      batch);







//    // ret = cublasSgemmBatched(handle,
//    //                   CUBLAS_OP_N,
//    //                   CUBLAS_OP_N,
//    //                   m,n,k,
//    //                   &alpha,
//    //                   d_Marray,  lda,
//    //                   d_Narray,  ldb,
//    //                   &beta,
//    //                   d_Parray,  ldc,
//    //                   batch);
//    cublasDestroy(handle);
//    if (ret == CUBLAS_STATUS_SUCCESS)
//    {
//    printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
//    }

//    show<<<1,1>>>(c_array[0], 16);
//    cudaMemcpy(c, d_c, sizeof(float)*16, cudaMemcpyDeviceToHost);
//    for(int i=0; i<16; i++) cout<<c[i]<<" "<<endl;


//    return 0;
// }

// #include <assert.h>
// #include <iostream>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
 
 
// using namespace std;
 
 
// //cuBLAS代码
// int main()
// {
// const float alpha = 1.0f;
// const float beta  = 0.0f;
// int m = 2, n = 4, k = 3;
 
 
// float A[6] = {1,2,3,4,5,6};
// float B[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
// float *C;
 
// float* d_A,*d_B, *d_C;
 
// C = (float*)malloc(sizeof(float)*8);
// cudaMalloc((void**)&d_A, sizeof(float)*6);
// cudaMalloc((void**)&d_B, sizeof(float)*12);
// cudaMalloc((void**)&d_C, sizeof(float)*8);
 
 
// cudaMemcpy(d_A, A, 6*sizeof(float),  cudaMemcpyHostToDevice);
// cudaMemcpy(d_B, B, 12*sizeof(float), cudaMemcpyHostToDevice);
 
// cublasHandle_t handle;
// cublasCreate(&handle);
// cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, 3, d_B, 4, &beta, d_C, 2);
// cublasDestroy(handle);
 
// cudaMemcpy(C, d_C, 8*sizeof(float), cudaMemcpyDeviceToHost);
 
// for(int i=0; i<8; i++)
// {
//    cout<<C[i]<<endl;
// }
 
// }