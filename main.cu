
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>

__global__ void add_vector(float *c, const float *a, const float *b, int width)
{
    int col = blockIdx.x * width + threadIdx.x;
	int row = blockIdx.y * width + threadIdx.y;
    c[col][row] = a[col][row] + b[col][row];
}

//for an array of size, fill it with a random float in [0,1]
void random_floats(float *A, int size)
{
	for (int i = 0; i < size; i++)
	{
		A[i] = ((float) rand()) / ((float) RAND_MAX);
	}
}


float my_abs(float a)
{
	if (a< 0)
		return -1 * a;
	return a;
}

//finds the machine epsilon for a float
float find_eps()
{
	float machEps = (float) 1.0;

        do {
           machEps /= (float) 2.0;
        }
        while ((float)(1.0 + machEps) != 1.0);

        return machEps;
}

int main()
{
	srand(time(0));
	const int SIZE = 1000;
	size_t bytes = 100 * 100 * sizeof(float);

	//initialize out pointers on host and device
	float *A, *B, *C;
	float *dA, *dB, *dC;

	//allocate vectors on host
	A = (float*)malloc(bytes);
	random_floats(A, SIZE);
	B = (float*)malloc(bytes);
	random_floats(B, SIZE);
	C = (float*)malloc(bytes);

	//alocate vectors on device
	cudaMalloc((void**)&dA, bytes);
	cudaMalloc((void**)&dB, bytes);
	cudaMalloc((void**)&dC, bytes);

	//copy the vector from host to device
	cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

	//perform the addition
	add_vector<<<1, SIZE >>>(dC, dB, dA);

	//copy our answer back to the cpu
	cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	//check for correctness
	float eps = find_eps();
	
	int worked = 0;
	for (int i = 0; i < SIZE; i++)
	{
		float rel_error = my_abs( ( (A[i] + B[i]) - C[i]) /C[i]);

		if ( rel_error > eps)
		{
			printf("messed up on index %d\n", i);
			printf("Calculation did not work\n");
			worked = 1;
			break;
		}
	}

	if (worked == 0)
		printf("Congrats, everyting worked!\n");
	

	//free up the host memory
	free(A);
	free(B);
	free(C);

}
