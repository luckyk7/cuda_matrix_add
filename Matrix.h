#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
	int rows;
	int cols;
	size_t bytes;
	float* data;

}Matrix;

#endif
