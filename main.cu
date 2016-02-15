/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernel.cu"
#include "support.h"

void cuda_error_check(int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[%s:%d] Cuda Error type: %s\n", __FILE__, line, cudaGetErrorString(err));
		exit(-1);
	}
}


int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d = 0, *B_d = 0, *C_d = 0;
    unsigned A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./sgemm                # All matrices are 1000 x 1000"
           "\n    Usage: ./sgemm <m>            # All matrices are m x m"
           "\n    Usage: ./sgemm <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
           "\n");
        exit(0);
    }

    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    /* Intializes random number generator */
    srand((unsigned) time(0));    


    A_h = (float*) malloc( sizeof(float)*A_sz );
    B_h = (float*) malloc( sizeof(float)*B_sz );
    C_h = (float*) malloc( sizeof(float)*C_sz );

	if (A_h == 0 || B_h == 0 || C_h == 0 )
	{
		FATAL("Unable to allocate memory on host");
		exit(-1);
	}

	for (unsigned int i=0; i < B_sz; i++) { B_h[i] = (float)(rand()%100)/100.00; }
	for (unsigned int i=0; i < A_sz; i++) { A_h[i] = (float)(rand()%100)/100.00; }
	for (unsigned int i=0; i < C_sz; i++) { C_h[i] = 0; }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", matArow, matAcol,
        matBrow, matBcol, matArow, matBcol);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

	cudaMalloc((void**)&A_d, A_sz * sizeof(float)); cuda_error_check(__LINE__);
	
	cudaMalloc((void**)&B_d, B_sz * sizeof(float)); cuda_error_check(__LINE__);

	cudaMalloc((void**)&C_d, C_sz * sizeof(float)); cuda_error_check(__LINE__);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

	cudaMemcpy(A_d, A_h, A_sz * sizeof(float), cudaMemcpyHostToDevice); cuda_error_check(__LINE__);

	cudaMemcpy(B_d, B_h, B_sz * sizeof(float), cudaMemcpyHostToDevice); cuda_error_check(__LINE__);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard sgemm interface ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f, \
		A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
	cudaMemcpy(C_h, C_d, C_sz * sizeof(float), cudaMemcpyDeviceToHost); cuda_error_check(__LINE__);


    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

	cudaFree(A_d); cuda_error_check(__LINE__);
	cudaFree(B_d); cuda_error_check(__LINE__);
	cudaFree(C_d); cuda_error_check(__LINE__);

    return 0;

}
