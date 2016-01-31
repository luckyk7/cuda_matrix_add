#include "Matrix.h"

Matrix create_matrix(int c, int r)
{
	Matrix temp;
	temp.rows = r;
	temp.cols = c;
	temp.bytes = r * c * sizeof(float);
	temp.data = (float*)malloc(temp.bytes);
	
	return temp;
}

static void allocate(float** m, size_t bytes)
{
	*m = (float*)malloc(bytes);
	if (m == 0)
	{
		puts("Error, could not allocate memory for matrix on host.");
		exit(0);
	}
}

void fill_rand_floats(Matrix matrix, int range)
{
	int i = 0;
	for (i = 0; i < matrix.cols* matrix.rows; i++)
	{
		matrix.data[i] = ((float)rand() / (float)RAND_MAX) * range;
	}
}
