/*=================================================================
 *=================================================================*/
#include <math.h>
#include "mex.h"

void SaveCellData(unsigned char Cell[], int Nx, int Ny, FILE *fom)
{
	unsigned long long int id;
	unsigned char* lnode = (unsigned char*)malloc(sizeof(unsigned char)*Nx);
	int* nn = (int*)malloc(sizeof(int)*Nx);
	unsigned char* in = (unsigned char*)malloc(sizeof(unsigned char)*Nx);
	int node0, mnode, inode;
	int lnn;
	char bufm[200];

	fprintf(fom, "%d %d\n", Nx, Ny);

	for(int j = 0; j < Ny; j++){
		for(int i = 0; i < Nx; i++){
			id = (unsigned long long int)Nx * (unsigned long long int)j + (unsigned long long int)i;
			lnode[i] = Cell[id];
		}

		node0 = lnode[0];
		mnode = 1;
		inode = 0;
		for(int i = 1; i < Nx; i++){
			if(lnode[i] == node0){
				mnode++;
				nn[inode] = mnode;
				in[inode] = node0;
			}
			else{
				nn[inode] = mnode;
				in[inode] = node0;
				mnode = 1;
				inode++;
				node0 = lnode[i];
			}
			nn[inode] = mnode;
			in[inode] = node0;
		}
		lnn = inode + 1;
		fwrite(&lnn, sizeof(int), 1, fom);
		fwrite(nn, sizeof(int), lnn, fom);
		fwrite(in, sizeof(unsigned char), lnn, fom);
	}
	return;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned char* Cell;
	char* CellName;
	FILE *fom;
	int Nx, Ny;

	CellName = mxArrayToString(prhs[0]);
	printf("%s\n", CellName);
	Cell = (unsigned char*)mxGetPr(prhs[1]);
	Nx = (int)mxGetScalar(prhs[2]);
	Ny = (int)mxGetScalar(prhs[3]);
	printf("%d  %d \n", Nx, Ny);
	
	fom = fopen(CellName, "wb");		// ボクセルデータ
	SaveCellData(Cell, Nx, Ny, fom);
	fclose(fom);
	return;
}
