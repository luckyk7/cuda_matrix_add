/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <math.h>

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

	// find what column and row we are in
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float product = 0.0f;
	//check if both indexes are in bounds
	if ( row < m && col < n )
	{
		for (int i = 0; i < k; i++)
		{
			product += A[row * k + i] * B[i * n + col];
		}
		C[row * n + col] = product;
	}
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread blocks

    //INSERT CODE HERE
	//works with square matrices
	dim3 dimGrid( (n+BLOCK_SIZE-1)/BLOCK_SIZE, (m+BLOCK_SIZE-1)/BLOCK_SIZE );
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);




    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

	mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);


}

