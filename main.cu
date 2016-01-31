#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include "Matrix.h"

#define TILE_WIDTH (int)16

//need to install lots of error checking here
float* transferToDevice(float* M_h, size_t size)
{
	float* M_d;

	cudaMalloc((void**)&M_d, size);
	cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
	
	return M_d;
}

void matrixMult(Matrix A_h, Matrix B_h, Matrix C_h)
{
	float* A_d = transferToDevice(A_h.data, A_h.bytes);
	float* B_d = transferToDevice(B_h.data, A_h.bytes);
	
	float* C_d;
	cudaMalloc((void**)&C_d, C_h.bytes );

	//delete this later
	int width = 20;
	dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);


	//launch the kernal here

	//copy the answer back to the host
	cudaMemcpy(C_h.data, C_d, )

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

}


int main()
{
	srand(time(0));

	Matrix A_h = create(16, 16);
	Matrix B_h = create(16, 16);
	if (A_h.rows == B_h.cols)
		Matrix C_h = create(A_h.cols, B_h.rows);
	else
	{
		printf("Can not multiply matrices of these dimensions.\n")
		return 0;
	}

	fill_rand_floats(A_h, 10);
	fill_rand_floats(B_h, 10);

	matrixMult(A_h, B_h, C_h);

	//delete allocated memory on the host
	free(A_h.data);
	free(B_h.data);
	free(C_h.data);
}
