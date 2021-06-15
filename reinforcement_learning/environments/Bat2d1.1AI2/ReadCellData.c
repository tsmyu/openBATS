#include "mex.h"

void ReadCellData(unsigned char Cell[], int Nx, int Ny, FILE *fim)
{
	int i, j, k, ix;
	unsigned long long int id;
	int node, n[30];
	char bufm[200];
	int* nn = (int*)malloc(sizeof(int)*Nx);
	unsigned char* in = (unsigned char*)malloc(sizeof(unsigned char)*Nx);
	int lnn;
	size_t size;

	for(int j = 0; j < Ny; j++){
		size = fread(&lnn, sizeof(int), 1, fim);
		size = fread(nn, sizeof(int), lnn, fim);
		size = fread(in, sizeof(unsigned char), lnn, fim);
		ix = 0;
		for(int i = 0; i < lnn; i++){
			for(int ii = 0; ii < nn[i]; ii++){
				id = (unsigned long long int)Nx * (unsigned long long int)j + (unsigned long long int)ix;
				Cell[id] = (unsigned char)in[i];
				ix++;
			}
		}
	}
	return;

}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned char* Cell;
	char* CellName;
	FILE *fim;
	int Nx, Ny;
	char buf[200];
	
	CellName = mxArrayToString(prhs[0]);
	printf("%s\n", CellName);
	
	fim = fopen(CellName, "rb");		// ボクセルデータ
	if(fgets(buf, sizeof(buf), fim) != NULL)
	sscanf(buf, "%d %d", &Nx, &Ny);			// 分割数だけ先に読み込む
	printf("%d  %d \n", Nx, Ny);
	plhs[0] = mxCreateNumericMatrix(Nx, Ny, mxUINT8_CLASS, mxREAL);
	Cell = (unsigned char*)mxGetPr(plhs[0]);
	
	ReadCellData(Cell, Nx, Ny, fim);
	fclose(fim);
}
