// 2����WE-FDTD�@ CUDA version
// 2017.01.07
// ver.0.01
// Takao Tsuchiya, Doshisha Univ.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <mpi.h>

//#define M_PI 3.14159265358979
#define HBD_TO_HBU 1
#define HBU_TO_HBD 10

// ���f���f�[�^
int iCell;					// ���f���^�C�v
char CellName[200] = {};			// �{�N�Z���t�@�C����
int Scheme, Boundary;		// ��@, ���E
float c0;
int3 Ndiv;					// x, y, z����������
int Nyorg;					// ����z����������
int Nreg, Nt, Ns;			// �̈敪����, �v�Z�X�e�b�v��, ��������
int Nobs;					// �ϑ��_��
struct Pnt{					// �ϑ��_���
	int x, y;
	float p;
};
Pnt Src, Rcv;				// �����E�󉹓_���W
Pnt* obs; 					// �ϑ��_���W
float* drv;					// �����g�`
int Nw, Srcw;				// �������ԕ�, ������
float freq;					// ���͎��g��
int Nd;						// �o�[�X�g�g��
float cfl, dl, dt, b;		// CFL, �T���v�����O���g��
float Ref[4], aref;			// ���E���˗�, �C�Ӌ��E���˗�
int iplane, ipn, iptime, iwave;		// �o�͕��ʁC���ʈʒu�C���ԊԊu�C�g�`�o��
int Nwave;					// �g�`�f�[�^�ꊇ�]�����ԃX�e�b�v��
float* pp;					// �������z�}
int istx, isty, ipts, ipte;	// �������z�}�Ԉ���
int Ndx, Ndy;				// �������z�}�̑傫��

// for GPU
int Ngpu;					// �g�pGPU��
int GpuId = -1;				// 1GPU�̂Ƃ���GPU ID
int3 Block;					// Block�T�C�Y
int Bblock = 256;
int Boff;					// Block offset
float mem;

// for MPI
int Nnode, inode;			// �m�[�h���C�����N
MPI_Status mpi_stat;		// MPI�X�e�[�^�X

// for moving
int iSource, iReceiver;		// ����/�󉹓_�ړ����@
char SPosName[200] = {}, RPosName[200] = {}; //�C�ӋO���ʒu�t�@�C����
float vs, vs0, as, t, angs;	// �������x, �����x�C�����x, �p�x
float vr, vr0, ar, angr;	// �󉹓_���x, �����x�C�����x, �p�x
float2 *SPos, *RPos;		// ����/�󉹓_�ʒu
int Nsrc, Nrcv;

#include "WE-FDTD_T.h"


int main(int argc, char* argv[]) 
{

	// MPI������
	int    resultlen;
	char   proc_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &Nnode);
	MPI_Comm_rank(MPI_COMM_WORLD, &inode);
	MPI_Get_processor_name( proc_name, &resultlen );

	// ���f���f�[�^�ǂݍ���
	float th1, th2, al1, al2, b1, b2, d1, d2;
	ReadModel(argc, argv);
	
	if(Boundary == 0){
		for(int ir = 0; ir < 4; ir++)
			Ref[ir] = ((1.0 + Ref[ir]) * cfl - (1.0 - Ref[ir])) / ((1.0 + Ref[ir]) * cfl + (1.0 - Ref[ir]));
	}
	if(Boundary == 1){
		th1 = 45.0 / 180.0 * M_PI;
		th2 = 45.0 / 180.0 * M_PI;
		al1 = 1.0 / cos(th1);
		al2 = 1.0 / cos(th2);
		b1 = (al1 * cfl - 1) / (al1 * cfl + 1);
		b2 = (al2 * cfl - 1) / (al2 * cfl + 1);
		d1 = 0.001;
		d2 = 0.001;
	}
	
	mem = (double)(Ndiv.x)*Ndiv.y*8 + (double)Nwave*Nobs*28 + Nobs*12*7;
	if(inode == 0){
		if(Scheme == 0) printf("SLF method\n");
		if(Scheme == 1) printf("IWB method\n");
		if(iCell > 0){
			printf(" Arbitrary model Model, file name: %s\n", CellName);
			mem += (double)(Ndiv.x)*Ndiv.y;
		}
		else
			printf(" Rectangular model\n");
			
		printf(" dl = %f(m), dt = %e(s), CFL = %f, b = %f\n", dl, dt, cfl, b);
		printf(" Nx = %d, Ny (%d) = %d, Nt = %d\n", Ndiv.x, Ndiv.y, Nyorg, Nt);
		printf(" Size: %f x %f(m) = %f(m^2), total time %f(s)\n", Ndiv.x*dl, Ndiv.y*dl, 
		Ndiv.x*dl*Ndiv.y*dl, Nt*dt);
		if(iSource == 0)
			printf(" Source  : fixed at (%d, %d) \n", Src.x, Src.y);
		if(iSource == 1)
			printf(" Source  : moving source, file name: %s \n", SPosName);
		if(iSource == 2)
			printf(" Source  : linear at v0 = %6.2f(m/s), a = %6.2f(m/s^2), theta = %3.0f(deg.)\n", vs0, as, angs);
		if(iReceiver == 0)
			printf(" Receiver: %d fixed observation points \n", Nobs);
		if(iReceiver == 1)
			printf(" Receiver: moving receiver, file name: %s \n", RPosName);
		if(iReceiver == 2)
			printf(" Receiver: linear at v0 = %6.2f(m/s), a = %6.2f(m/s^2), theta = %3.0f(deg.)\n", vr0, ar, angr);
	}

	// �f�o�C�X������
	int Num_gpu = 0;
	cudaGetDeviceCount(&Num_gpu);
	cudaDeviceProp dev;
	MPI_Barrier(MPI_COMM_WORLD);
	if(inode == 0){
		printf(" %d MPI nodes are used, %d GPUs are used in each node\n", Nnode, Ngpu);
		printf(" No. of divided regions = %d, Total GPUs used = %d\n", Nreg, Ngpu*Nnode);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	printf(" %s: Node = %d, %d GPUs found, %d GPUs are used\n", proc_name, inode, Num_gpu, Ngpu);
	if( Num_gpu < Ngpu ) {
		printf("error:: Proc %s, rank %d, %d GPUs found, %d GPUs are used\n", proc_name, inode, Num_gpu, Ngpu);
		MPI_Abort( MPI_COMM_WORLD, 1 );
		exit(1);
	}
	if(inode == 0){
		cudaGetDeviceProperties(&dev, 0);
		printf(" Global Memory Usage: %f (GB), Total Global Memory %f (GB)\n\n", mem/1024./1024./1024., 
			Nnode*Ngpu*dev.totalGlobalMem/1024./1024./1024.);
		if(mem > Nnode*Ngpu*dev.totalGlobalMem){
			printf(" Momory over!!\n");
			exit(1);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	// 1�u���b�NBlock_x*Block_y�̃X���b�h���ŕ���v�Z
	MPI_Barrier(MPI_COMM_WORLD);
	int Nydiv = Ndiv.y / Nreg;
	int Nygpu = Nydiv + Block.y;
	int bdx = Ndiv.x / Block.x;
	int bdy = Nygpu / Block.y;
	if(inode == 0){
		printf(" x division for each GPU: %d\n", Nygpu);
		printf(" Block: %d, %d ", Block.x, Block.y);
		printf(" Block Size: %d * %d, Thread Size: %d\n", bdx, bdy, Block.x*Block.y);
	}


	// �z�X�g���float�^�̃��������m�ۂ���
	int Nbm;
	Nbm = Ndiv.x;
	float* dpt = (float*) malloc(sizeof(float)*Boff*Nbm*Ngpu);		// OMP�p���E�f�[�^(��w)
	float* dpb = (float*) malloc(sizeof(float)*Boff*Nbm*Ngpu);		// OMP�p���E�f�[�^(���w)
	float* dppt = (float*) malloc(sizeof(float)*Boff*Nbm*Ngpu);		// OMP�p���E�f�[�^(��w)
	float* dppb = (float*) malloc(sizeof(float)*Boff*Nbm*Ngpu);		// OMP�p���E�f�[�^(���w)
	float* mpt = (float*) malloc(sizeof(float)*Boff*Nbm);			// MPI�p���E�f�[�^(��w)
	float* mppt = (float*) malloc(sizeof(float)*Boff*Nbm);			// MPI�p���E�f�[�^(��w)
	float* mpb = (float*) malloc(sizeof(float)*Boff*Nbm);			// MPI�p���E�f�[�^(���w)
	float* mppb = (float*) malloc(sizeof(float)*Boff*Nbm);			// MPI�p���E�f�[�^(���w)
	for(int i = 0; i < Boff*Nbm*Ngpu; ++i) {
		dpt[i] = dpb[i] = dppt[i] = dppb[i] = 0;
	}
	for(int i = 0; i < Boff*Nbm; i++){
		mpt[i] = mppt[i] = mpb[i] = mppb[i] = 0;
	}


	// �ϑ��g�`�p
	Nwave = 100;
	float* wave  = (float*) malloc(sizeof(float)*Nwave*Nobs*3);	// �ϑ��_�����g�`
	char WaveName[200] = {};			// �g�`�f�[�^�t�@�C����
	char WaveNamebin[200] = {};			// �g�`�f�[�^�t�@�C����
	for(int i = 0; i < Nwave*Nobs*3; i++)
		wave[i] = 0;
		
	char tmp[10];
	if(Src.p == 0)
		strcat(WaveName, "wave0_x");
	if(Src.p == 1)
		strcat(WaveName, "waveX_x");
	if(Src.p == 2)
		strcat(WaveName, "waveY_x");
	sprintf(tmp, "%d", Src.x);
	strcat(WaveName, tmp);
	strcat(WaveName, "_y");
	sprintf(tmp, "%d", Src.y);
	strcat(WaveName, tmp);
	strcat(WaveNamebin, WaveName);
	strcat(WaveNamebin, ".bin");
	strcat(WaveName, ".csv");
//	FILE *fp2 = fopen(WaveName,"w");
	FILE *fpb = fopen(WaveNamebin,"wb");
	fwrite(&Nt, sizeof(int), 1, fpb);


	// �������z�}�p�z��
	pp = (float*) malloc(sizeof(float)*Ndiv.x*Ndiv.y);

	// OMP�J�n
	printf(" Calculation start!\n");
	MPI_Barrier(MPI_COMM_WORLD);
	omp_set_num_threads(Ngpu);			// create as many CPU threads as there are CUDA devices
	cudaEvent_t start,stop;
	#pragma omp parallel
	{
		unsigned int cpu_id = omp_get_thread_num();
		unsigned int Nthreads = omp_get_num_threads();

		int gpu_id;
		if(Nreg == 1 && GpuId < Num_gpu)
			cudaSetDevice(GpuId);	// "% num_gpus" allows more CPU threads than GPU devices
		else
			cudaSetDevice(cpu_id % Ngpu);	// "% num_gpus" allows more CPU threads than GPU devices
		cudaGetDevice(&gpu_id);
		printf(" Node: %d, thread: %d uses CUDA device %d\n", inode, cpu_id, gpu_id);

		unsigned char* Cell;
		unsigned long long *Bid;
		unsigned short *Bnode;
		int Nbnd = 0;
		unsigned long long id, Nem;
		int iReg, nbx;
		float* pobs  = (float*) malloc(sizeof(float));			 	// �ϑ��_����

		Nem = (unsigned long long)Ndiv.x * Nygpu;
		if(Ngpu == 1)
			iReg = inode * Ngpu;
		else
			iReg = inode * Ngpu + gpu_id;

		float* hwave = (float*)malloc(sizeof(float)*Nwave*Nobs*3);	// �ϑ��_�����g�`
		for(int i = 0; i < Nwave*Nobs*3; i++)
			hwave[i] = 0;

		if(iCell > 0){
			Cell  = (unsigned char*)malloc(sizeof(unsigned char)*Nem);
			for(int j = 0; j < Nygpu; j++){
				for(int i = 0; i < Ndiv.x; i++){
					id = (unsigned long long)(Ndiv.x * j + i);
					Cell[id] = 0;
				}
			}
	
			Nbnd = ReadCell(Cell, iReg, Nydiv, Nygpu);

			nbx = Nbnd / Bblock + 1;
			Nbnd = Bblock * nbx;

			Bid = (unsigned long long*)malloc(sizeof(unsigned long long)*Nbnd);
			Bnode = (unsigned short*)malloc(sizeof(unsigned short)*Nbnd);
			for(int i = 0; i < Nbnd; i++){
				Bid[i] = 0;
				Bnode[i] = 0;
			}
			SetBoundary2d(Cell, Bid, Bnode, Nygpu, Nem);
		}


		// �f�o�C�X��Ƀ��������m�ۂ���
		float *dp, *dpp, *tmp, *dRef, *dp2, *dux, *duy;
		unsigned char *dCell;
		unsigned long long *dBid;
		unsigned short *dBnode;
		float *dwave, *ddrv;
		Pnt *dobs;				// �ϑ��_���W
//		int3 *dobs;				// �ϑ��_���W
		
		cudaMalloc((void**) &dp,    sizeof(float)*Nem);			// ����
		cudaMalloc((void**) &dpp,   sizeof(float)*Nem);			// 1�X�e�b�v�O����
		cudaMalloc((void**) &dp2,   sizeof(float)*6*(Ndiv.x+Ndiv.y));			// 1�X�e�b�v�O����
		cudaMalloc((void**) &dobs,  sizeof(Pnt)*Nobs*5);			// �ϑ��g�`�p
		cudaMalloc((void**) &dwave, sizeof(float)*Nwave*Nobs*3);	// �ϑ��g�`�p
		cudaMalloc((void**) &dux, sizeof(float)*Nobs);	// �ϑ��g�`���q���xx�p
		cudaMalloc((void**) &duy, sizeof(float)*Nobs);	// �ϑ��g�`���q���xy�p
		cudaMalloc((void**) &ddrv, sizeof(float)*Nt);	// �ϑ��g�`�p
		cudaMalloc((void**) &dRef, sizeof(float)*4);			// ���ˌW��
		if(iCell > 0){
			cudaMalloc((void**) &dCell, sizeof(unsigned char)*Nem);			// �`��f�[�^
			cudaMalloc((void**) &dBid, sizeof(unsigned long long)*Nbnd);	// ���E����id�f�[�^
			cudaMalloc((void**) &dBnode, sizeof(unsigned short)*Nbnd);		// ���E���ˌW���f�[�^
		}

		// �f�o�C�X�������̏�����
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);
		cudaMemset(dp,   0, sizeof(float)*Nem);
		cudaMemset(dpp,  0, sizeof(float)*Nem);
		cudaMemset(dp2,  0, sizeof(float)*6*(Ndiv.x+Ndiv.y));
		cudaMemcpy(dobs, obs, sizeof(Pnt)*Nobs, cudaMemcpyHostToDevice);
		cudaMemset(dwave,0, sizeof(float)*Nwave*Nobs*3);
		cudaMemset(dux,0, sizeof(float)*Nobs);
		cudaMemset(duy,0, sizeof(float)*Nobs);
		cudaMemcpy(ddrv, drv, sizeof(float)*Nt, cudaMemcpyHostToDevice);
		cudaMemcpy(dRef, Ref, sizeof(float)*4, cudaMemcpyHostToDevice);
		if(iCell > 0){
			cudaMemcpy(dCell, Cell, sizeof(unsigned char)*Nem, cudaMemcpyHostToDevice);
			cudaMemcpy(dBid, Bid, sizeof(unsigned long long)*Nbnd, cudaMemcpyHostToDevice);
			cudaMemcpy(dBnode, Bnode, sizeof(unsigned short)*Nbnd, cudaMemcpyHostToDevice);
		}

		dim3 grid(bdx, bdy, 1);
		dim3 threads(Block.x, Block.y, 1);
		dim3 gridb(nbx, 1, 1);
		dim3 threadsb(Bblock, 1, 1);
		dim3 gridw(1, 1, 1);
		dim3 threadsw(Nobs, 1, 1);
		
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);

		// communication test
		if(Nnode > 1){
			for(int i = 0; i < 20; ++i){
				#pragma omp single
				{
					if(inode == 0){
						MPI_Send((void*)(dpt+Nbm*(Ngpu-1)), Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, MPI_COMM_WORLD);
					}else if(inode == Nnode -1){
						MPI_Recv((void*)mpt, Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
					}else{
						MPI_Sendrecv((void*)(dpt+Nbm*(Ngpu-1)), Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, 
									(void*)mpt, Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
					}
				}
			}
		}
		cudaError_t err = cudaGetLastError();
		printf("cuda debug:: line:%d rank:%d gpu:%d msg:%s\n", __LINE__, inode, gpu_id, cudaGetErrorString(err));

		// for moving source
		float2 XY, Psrc, dm;
		int3 Msrc;
		int itt;
		float rads = angs / 180. * M_PI;
		float dr;
		float4 Driv;
		Driv.x = Driv.y = Driv.w = 0.0;
		float phi;
		phi   = M_PI / 180. * Src.p;	// �����p
		
		int Nyoff, Nys, Nye;
		Nyoff = Nydiv * iReg - Boff;
		Nys = Nyoff;
		Nye = Nyoff + Nygpu - 1;
		if(iReg == 0) Nys = 0;
		if(iReg == Nreg-1) Nye = Nyorg - 1;
//		if(iReg == Nreg-1) Nye = Ndiv.y - 1;
		
		if(iReg == 0) printf("Nydiv: %d  Nygpu: %d\n", Nydiv, Nygpu);
		printf("Nyoff: %d\n", Nyoff);
		printf("Nys: %d  Nye: %d\n", Nys, Nye);

		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);
		if(inode == 0 && (Nreg == 1 || gpu_id == 0)){
			// �^�C�}�[���쐬���Čv���J�n
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start,0);
		}

		// ���ԃ��[�v
		for(int it = 0; it < Nt; it++){

			itt = it % Nwave;
			if((itt == 0 || it == Nt - 1) && inode == 0){
				if(Ngpu == 1 || gpu_id == 0)
					printf("step: %d\n", it);
				err = cudaGetLastError();
				if(err != 0)
					printf("cuda debug:: line:%d rank:%d gpu:%d msg:%s\n", __LINE__, inode, gpu_id, cudaGetErrorString(err));
			}

			// �����v�Z
			t = it * dt;

			dr = 0;
			XY.x = 0;
			XY.y = 0;
			if(Nsrc > it) dr = drv[it];
			
			if(iSource == 1 && Nsrc > it){
				Psrc.x = SPos[it].x / dl;
				Psrc.y = SPos[it].y / dl;
			}
			if(iSource == 0){
				Msrc.x = Src.x;
				Msrc.y = Src.y;
				Msrc.z = Src.p;
				Driv.w = drv[it];
				if(phi == 0.0){
					Driv.w = drv[it];
				}
				else{
					Driv.x += drv[it] * cfl;
					Driv.y = Driv.x;
				}
			}
			if(iSource == 2){
				Psrc.x = Src.x + (as * t * t / 2. + vs0 * t) / dl * cos(rads);
				Psrc.y = Src.y + (as * t * t / 2. + vs0 * t) / dl * sin(rads);
			}
//			Msrc.x = int(Psrc.x);
//			Msrc.y = int(Psrc.y);
			if(Msrc.x > Ndiv.x-1) Msrc.x = Ndiv.x;
			if(Msrc.y > Ndiv.y-1) Msrc.y = Ndiv.y;
			dm.x = Psrc.x - Msrc.x;
			dm.y = Psrc.y - Msrc.y;
			XY.x = dm.x * 2 - 1;
			XY.y = dm.y * 2 - 1;
//			printf("%d %d %f %f \n", Msrc.x, Msrc.y, XY.x, XY.y);

			ABC_Higdon_store<<<grid, threads>>>(dpp, dp2, Ndiv.x, Nyorg, Nyoff, Nys, Nye);

			// �̈�v�Z
			cudaDeviceSynchronize();
			#pragma omp barrier
			#pragma omp single
			MPI_Barrier(MPI_COMM_WORLD);
			
			if(Scheme == 0){
//				SLF<<<grid, threads>>>(dp, dpp, Ndiv.x, Ndiv.y, Src, drv[it], cfl, Nyoff, Nys, Nye, sx, dvp, dvm, g);
			}
			else{
//				IWBmoving<<<grid, threads>>>(dp, dpp, Ndiv.x, Ndiv.y, Msrc, dr, cfl, Nyoff, Nys, Nye, b, XY, iCell, dCell);
				IWBfixed<<<grid, threads>>>(dp, dpp, Ndiv.x, Ndiv.y, Msrc, Driv, cfl, Nyoff, Nys, Nye, b, iCell, dCell);
			}
			cudaDeviceSynchronize();
			tmp = dpp; dpp = dp; dp = tmp;

			if(Boundary == 0)
				ABC_Mur<<<grid, threads>>>(dp, dpp, Ndiv.x, Ndiv.y, dRef, Nyoff);
			else{
				ABC_Higdon_plane<<<grid, threads>>>(dp, dpp, dp2, Ndiv.x, Nyorg, dRef, Nyoff, Nys, Nye, b1, b2, d1, d2);
				ABC_Higdon_corner<<<grid, threads>>>(dp, dpp, dp2, Ndiv.x, Nyorg, dRef, Nyoff, Nys, Nye, b1, b2, d1, d2);
			}
			cudaDeviceSynchronize();
			#pragma omp barrier
			#pragma omp single
			MPI_Barrier(MPI_COMM_WORLD);
			// ���E�v�Z
			if(iCell > 0 && Nbnd > 0){

				CE_boundary_Plane2d<<<gridb, threadsb>>>(dp, dpp, aref, dCell, dBid, dBnode, Ndiv.x, cfl, Nbnd);
				cudaDeviceSynchronize();
				#pragma omp barrier
				#pragma omp single
				MPI_Barrier(MPI_COMM_WORLD);

				CE_boundary_Edge2d<<<gridb, threadsb>>>(dp, dpp, aref, dCell, dBid, dBnode, Ndiv.x, cfl, Nbnd);
				cudaDeviceSynchronize();
				#pragma omp barrier
				#pragma omp single
				MPI_Barrier(MPI_COMM_WORLD);

			}
			if(Nreg > 1)
				ExcangeBoundary(dpt, dppt, dpb, dppb, dp, dpp, mpt, mppt, mpb, mppb, Nem, Nbm, Nydiv, inode, gpu_id, Nygpu);

			// �ϑ��_�����擾
			if(iwave > 0){
				if(iReceiver == 0) ObssEcho<<<gridw, threadsw>>>(dp, Ndiv, Nydiv, Block, iReg, dobs, dwave, dux, duy, cfl, Nwave, itt);
//				if(iReceiver == 0) WaveObss<<<gridw, threadsw>>>(dp, Ndiv, Nydiv, Block, iReg, dobs, dwave, du, cfl, Nwave, itt);
				if(iReceiver > 0) MovingReceiver(dp, wave, Ndiv, Nydiv, Block, iReg, Boff, Rcv, RPos, it);
				cudaDeviceSynchronize();
			}
			#pragma omp barrier
			#pragma omp single
			MPI_Barrier(MPI_COMM_WORLD);

			if(iwave > 0){
				if(itt == Nwave - 1 || it == Nt - 1){
//					SaveWave(dwave, hwave, wave, it, gpu_id, Nydiv, fp2);
					SaveWaveBin(dwave, hwave, wave, it, gpu_id, Nydiv, fpb);
					cudaDeviceSynchronize();
					#pragma omp barrier
					#pragma omp single
					MPI_Barrier(MPI_COMM_WORLD);
				}
			}
			cudaDeviceSynchronize();
			#pragma omp barrier
			#pragma omp single
			MPI_Barrier(MPI_COMM_WORLD);

			// �������z�}�ۑ�
			if(iplane > 0 && it % iptime == 0 && it > 0){
				ipn = 1;
				Ndx = ceil((double)Ndiv.x / istx);
				Ndy = ceil((double)Ndiv.y / isty);
				save_cross_section(dp, pp, Ndiv, inode, gpu_id, it, Nydiv, Boff, iReg, ipn, istx, isty);
				#pragma omp barrier
				#pragma omp single
				MPI_Barrier(MPI_COMM_WORLD);
			}

		}
		
		//�^�C�}�[���~�������������Ԃ�\��
		if(inode == 0 && (Nreg == 1 || gpu_id == 0)){
			float elapsed_time = 0.0;
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed_time, start, stop);
			printf("time: %f s\n", elapsed_time / 1000.);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}

	}
	
	MPI_Barrier(MPI_COMM_WORLD);
//	fclose(fp2);
	fclose(fpb);
	MPI_Finalize();

    return 0;

}

