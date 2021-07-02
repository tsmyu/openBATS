void ReadModel(int argc, char* argv[])
{
	char buf[200], SrcName[200];
	float ang;
	int dum;

//	SrcName = (char*) malloc(sizeof(char)*20); 			// �����g�`�t�@�C����
	FILE *fi  = fopen("input.dat","r");
	if(fi == NULL){
		printf("error:: No input file!\n");
		MPI_Abort( MPI_COMM_WORLD, 1 );
		exit(1);
	}

	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d", &dum);							// 1�s�ڂ̓R�����g
	
	Scheme = 0;
	Boundary = 0;
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d %d %d", &iCell, &Scheme, &Boundary);	// �C�Ӌ��E(0:��`�C1:�C��)�C��@(0:SLF, 1:IWB), ���E����(0:Mur1��,1:Higdon2��)

	if(fgets(buf, sizeof(buf), fi) != NULL)
	sscanf(buf, "%d %d %s", &Ndiv.x, &Ndiv.y, CellName);	// ������, �Z���f�[�^�t�@�C����

	if(iCell > 0){										// �C�Ӄ{�N�Z�����f��
		FILE *fim  = fopen(CellName, "rb");
		if(fim == NULL){
			if(inode == 0) printf("error:: No CELL file!\n");
			MPI_Abort( MPI_COMM_WORLD, 1 );
			exit(1);
		}
		if(fgets(buf, sizeof(buf), fim) != NULL)
		sscanf(buf, "%d %d", &Ndiv.x, &Ndiv.y);			// ������������ɓǂݍ���
		fclose(fim);
	}
	else{												// ��`���f��
	}
	Block.x = 4;
	Block.y = 4;
	Boff = Block.y / 2;
//	printf("%d %d\n", Ndiv.x, Ndiv.y);

	for(int ir = 0; ir < 4; ir++)
		Ref[ir] = 0;
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%f %f %f %f %f", &Ref[0], &Ref[1], &Ref[2], &Ref[3], &aref);
	
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%f %f %f", &cfl, &dl, &c0);				// ���C��t, c0
	if(Scheme == 0){
		printf("2D SLF method\n");
		if(cfl == 0) cfl = 1. / sqrt(2);
	}
	if(Scheme == 1){
		printf("2D IWB method\n");
		b = 1.0 / 4.0;
		if(cfl == 0) cfl = 1.;
	}

	dt = cfl / c0 * dl;
	
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d", &Nt);								// �v�Z�X�e�b�v��
	
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d %s", &iSource, SPosName);			// �ړ��`��, �������W�t�@�C����

	vs0 = 0;
	as = 0;
	angs = 0;
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d %d %f %f %f %f", &Src.x, &Src.y, &Src.p, &vs0, &as, &angs);	// �����ʒu����, �������x�C�����x�C����
	if(argc > 1){
		Src.x = atoi(argv[1]);
		Src.y = atoi(argv[2]);
		Src.p = atoi(argv[3]);
//		printf("arg1: %d  arg2: %d\n", Src.x, Src.y);
	}

	int it = 0;
	Nsrc = Nt;
	if(iSource == 1){
		SPos = (float2*) malloc(sizeof(float2)*Nt); 		// �ϑ��_���W

		FILE *fpo = fopen(SPosName, "r");
		if(fpo == NULL){
			if(inode == 0) printf("error:: No source position file!\n");
			MPI_Abort( MPI_COMM_WORLD, 1 );
			exit(1);
		}
		while(fgets(buf, sizeof(buf), fpo) != NULL && it < Nt){
			sscanf(buf, "%f,%f", &SPos[it].x, &SPos[it].y);
//			printf("%f %f \n", SPos[it].x, SPos[it].y);
			it++;
		}
		Nsrc = it;
		fclose(fpo);
	}

	if(Src.x > Ndiv.x || Src.y > Ndiv.y){
		printf("error:: Invalid source location (%d, %d)\n", Src.x, Src.y);
		MPI_Abort( MPI_COMM_WORLD, 1 );
		exit(1);
	}
	
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%f %d %s", &freq, &Nd, SrcName);					// �������g���C������(0: �A��, -1: �C���p���X)
	
	Nw = 0;
	drv = (float*) malloc(sizeof(float)*Nt); 			// �����g�`
	for(int it = 0; it < Nt; it++){
		drv[it] = 0.;
	}
	if(freq == -1){										// �����t�@�C������g�`����
		printf("%s\n", SrcName);
		FILE *fi2  = fopen(SrcName, "r");
		for(int it = 0; it < Nt; it++){
			if(fgets(buf, sizeof(buf), fi2) != NULL){
				sscanf(buf, "%f", &drv[it]);
//				printf("%f\n", drv[it]);
				Nw++;
			}
		}
		fclose(fi2);
	}
	else if(freq == 0){								// �C���p���X
		drv[0] = 1.;
		Nw = 1;
	}
	else{
		if(Nd == 0){										// �A���g
			for(int it = 0; it < Nt; it++){
				drv[it] = sin(2 * 3.1415926 * freq * it * dt);
			}
			Nw = Nt;
		}
		else if(Nd == 1){		// �����K�E�X
			double wl = 2.0 / freq / dt;
			Nw = wl;
			if(Nw > Nt) Nw = Nt;
			for(int it = 0; it < 2*Nw; it++){
				drv[it] = -8*(it-Nw)*exp(-12.0*(it-wl)*(it-wl)/wl/wl)/wl;
	//			printf("%f\n", drv[it]);
			}
		}
		else{					// �g�[���o�[�X�g
			Nw = 1.0 / freq / dt * Nd;
			if(Nw > Nt) Nw = Nt;
			for(int it = 0; it < Nw; it++){
				drv[it] = sin(2 * 3.1415926 * freq * it * dt);
			}
		}
	}

	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d %s", &iReceiver, RPosName);			// �ړ��`��, �󉹓_���W�t�@�C����

	float sep = 0.01;
	float obsx, obsy;
	Nobs = 37;
	if(iReceiver == 0){
		obs  = (Pnt*) malloc(sizeof(Pnt)*Nobs); 			// �ϑ��_���W
		for(int io = 0; io < Nobs; io++){
			ang = io * 10 * 3.1415926 / 180.;
			obsx = (Src.x) * dl + sep * cos(ang);
			obsy = (Src.y) * dl + sep * sin(ang);
			obs[io].x = obsx / dl;
			obs[io].y = obsy / dl;
			obs[io].p = 0;
			printf("%d %d %f\n", obs[io].x, obs[io].y, obs[io].p);
		}
	}
	it = 0;
	Nrcv = Nt;
	if(iReceiver == 1){
		RPos = (float2*) malloc(sizeof(float2)*Nt); 		// �ϑ��_���W

		FILE *fpo = fopen(RPosName, "r");
		if(fpo == NULL){
			if(inode == 0) printf("error:: No receiver position file!\n");
			MPI_Abort( MPI_COMM_WORLD, 1 );
			exit(1);
		}
		while(fgets(buf, sizeof(buf), fpo) != NULL && it < Nt){
			sscanf(buf, "%f,%f", &RPos[it].x, &RPos[it].y);
//			printf("%f %f \n", RPos[it].x, RPos[it].y);
			it++;
		}
		Nrcv = it;
		fclose(fpo);
	}
	if(iReceiver == 2){
		if(fgets(buf, sizeof(buf), fi) != NULL);
		sscanf(buf, "%d %d %f %f %f %f", &Rcv.x, &Rcv.y, &Rcv.p, &vr0, &ar, &angr);	// �����ʒu����, �������x�C�����x�C����
	}

	if(fgets(buf, sizeof(buf), fi) != NULL)
		sscanf(buf, "%d %d", &iplane, &iwave);		// �R���^�[�ʁC�g�`�o��
	if(fgets(buf, sizeof(buf), fi) != NULL)
		sscanf(buf, "%d %d %d %d %d", &iptime, &istx, &isty, &ipts, &ipte);								// �R���^�[�Ԉ���
	if(ipte == 0) ipte = Nt;
	
	if(fgets(buf, sizeof(buf), fi) != NULL);
	sscanf(buf, "%d %d", &Ngpu, &GpuId);							// GPU��/�m�[�h, 1GPU�̂Ƃ���GPU ID
	
	if(Ngpu < 1){
		if(inode == 0) printf("error:: Invalid number of GPUs.: %d\n", Ngpu);
		MPI_Abort( MPI_COMM_WORLD, 1 );
		exit(1);
	}
	Nreg = Nnode * Ngpu;
		
	// �����`�F�b�N
	int Fx, Fy;
	Nyorg = Ndiv.y;
	if(Ndiv.y % (Nreg * Block.y) != 0)
		Ndiv.y = Nreg * Block.y * ((int)(Ndiv.y / (Nreg * Block.y)) + 1);
	Fx = Ndiv.x % Block.x;
	Fy = (int)(Ndiv.y / Nreg) % Block.y;
	if(Fx + Fy > 0){
		if( inode == 0 ){
			printf("error:: Invalid number of divisions.\n" );
			if(Fx > 0) printf("Nx: %d is invalid\n", Ndiv.x);
			if(Fy > 0) printf("Ny: %d is invalid\n", Ndiv.y);
		}
		MPI_Abort( MPI_COMM_WORLD, 1 );
		exit(1);
	}

}


int ReadCell(unsigned char* Cell, int iReg, int Nydiv, int Nygpu)
{
	unsigned long long id;
	int Nbnd = 0, dum;
	char bufm[200];
		
	FILE *fim  = fopen(CellName, "rb");		// �{�N�Z���f�[�^
	if(fim == NULL){
		printf("error:: No CELL file!\n");
		exit(1);
	}

	// �Z���f�[�^�ǂݍ���
	if(iReg == 0) printf(" Reading cell data: %s\n", CellName);
	int* nn = (int*)malloc(sizeof(int)*Ndiv.x);
	unsigned char* in = (unsigned char*)malloc(sizeof(unsigned char)*Ndiv.x);
	int lnn = 0;
	size_t size;

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);
	int ix, iy;
	int Nyoff, Nys, Nye;
	
	if(fgets(bufm, sizeof(bufm), fim) != NULL)
	sscanf(bufm, "%d", &dum);
//	printf("%d %d\n", dum, dum);
	
	Nyoff = Nydiv * iReg - Boff;
	Nys = Nyoff;
	Nye = Nyoff + Nygpu - 1;
	if(iReg == 0) Nys = 0;
	if(iReg == Nreg-1) Nye = Nyorg - 1;
		
//	printf("Nydiv: %d  Nygpu: %d\n", Nydiv, Nygpu);
//	printf("Nyoff: %d\n", Nyoff);
//	printf("Nys: %d  Nye: %d\n", Nys, Nye);

	for(int j = 0; j < Nys; j++){
		size = fread(&lnn, sizeof(int), 1, fim);
		size = fread(nn, sizeof(int), lnn, fim);
		size = fread(in, sizeof(unsigned char), lnn, fim);
	}

	for(int j = 0; j < Nygpu; j++){
		iy = j + Nyoff;
		
		if(iy < Nyorg){
			if(iy >= Nys && iy <= Nye){

				size = fread(&lnn, sizeof(int), 1, fim);
//				printf("%u\n", lnn);
				size = fread(nn, sizeof(int), lnn, fim);
				size = fread(in, sizeof(unsigned char), lnn, fim);
				if(size == 0){
					printf("error:: Illegal boundary data!! at iReg=%d\n", iReg);
					exit(1);
				}
				ix = 0;
				for(int i = 0; i < lnn; i++){
					for(int ii = 0; ii < nn[i]; ii++){
						id = (unsigned long long)Ndiv.x * j + ix;
						Cell[id] = in[i];
						ix++;
						if(Cell[id] > 1) Nbnd++;		// ���E�����f�[�^��
					}
				}
			}
		}
	}
	fclose(fim);

//	if(Nbnd < 1){
//		printf("error:: No boundary data!!\n");
//		exit(1);
//	}
	return Nbnd;

}


void SetBoundary2d(unsigned char *Cell, unsigned long long *Bid, unsigned short *Bnode, int Nygpu, unsigned long long Nem)
{
	unsigned long long id = 0;
	int node, n[30];
	int Nx;
	int bx, by, bb, bid = 0;

	Nx = Ndiv.x;
	for(int j = 1; j < Nygpu-1; j++){
		for(int i = 1; i < Nx-1; i++){
			id = (unsigned long long)Nx * j + i;
			node = 0;
			if(Cell[id] > 1){
				n[0] = Cell[id-1   ];
				n[1] = Cell[id  +Nx];
				n[2] = Cell[id+1   ];
				n[3] = Cell[id  -Nx];
				n[4] = Cell[id-1-Nx];
				n[5] = Cell[id-1+Nx];
				n[6] = Cell[id+1+Nx];
				n[7] = Cell[id+1-Nx];

				bx = by = 0;

				if(n[0] == 1) {bx = 1; bb = 1;}
				if(n[1] == 1) {by = 2; bb = 1;}
				if(n[2] == 1) {bx = 2; bb = 1;}
				if(n[3] == 1) {by = 1; bb = 1;}

				if(bb == 0){
					bb = 0;
						 if(n[4] == 1){bx = 1; by = 1;}
					else if(n[5] == 1){bx = 1; by = 2;}
					else if(n[6] == 1){bx = 2; by = 2;}
					else if(n[7] == 1){bx = 2; by = 1;}

				}

				if((bx+by) == 0){
					node = 0;
				}
				else{
					node = 9 * bb + 3 * by + bx;
				}
				if(node > 0){
					id = (unsigned long long)Nx * j + i;
					Bid[bid] = id;
					Bnode[bid] = (unsigned short)node;
					++bid;
					if(id > Nem) printf("%d %d: %lld\n", i, j, id);
				}
			}
		}
	}
	return;
}


__global__ void IWBfixed(float* dp, float* dpp, int Nx, int Ny, 
		int3 Src, float4 Driv, float cfl, int Nyoff, int Nys, int Nye, float b, int 
		iCell, unsigned char* dCell){
	
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;

	float d1, d2;
	unsigned char ce;

	d1 = cfl * cfl * (1 - 2 * b);
	d2 = cfl * cfl * b;
	ce = 1;
	if(iCell > 0) ce = dCell[id];
	
	if(ce == 1){
		if(ix > 0 && ix < Nx-1 && iy > Nys && iy < Nye){
			dpp[id] = (dp[id+1] + dp[id-1] + dp[id+Nx] + dp[id-Nx] - 4 * dp[id]) * d1
					+ (dp[id+1+Nx] + dp[id+1-Nx] + dp[id-1+Nx] + dp[id-1-Nx] - 4 * dp[id]) * d2
					+ 2 * dp[id] - dpp[id];

			if(Src.z == 0){
				if(ix == Src.x   && iy == Src.y  ) dpp[id] += Driv.w;
				if(ix == Src.x+1 && iy == Src.y  ) dpp[id] += Driv.w;
				if(ix == Src.x   && iy == Src.y+1) dpp[id] += Driv.w;
				if(ix == Src.x+1 && iy == Src.y+1) dpp[id] += Driv.w;
			}
			if(Src.z == 1){
				if(ix == Src.x+1 && iy == Src.y  ) dpp[id] += Driv.x / 2.;
				if(ix == Src.x+2 && iy == Src.y  ) dpp[id] += Driv.x / 2.;
				if(ix == Src.x+1 && iy == Src.y+1) dpp[id] += Driv.x / 2.;
				if(ix == Src.x+2 && iy == Src.y+1) dpp[id] += Driv.x / 2.;

				if(ix == Src.x-1 && iy == Src.y  ) dpp[id] -= Driv.x / 2.;
				if(ix == Src.x   && iy == Src.y  ) dpp[id] -= Driv.x / 2.;
				if(ix == Src.x-1 && iy == Src.y+1) dpp[id] -= Driv.x / 2.;
				if(ix == Src.x   && iy == Src.y+1) dpp[id] -= Driv.x / 2.;
			}
			if(Src.z == 2){
				if(ix == Src.x   && iy == Src.y+1) dpp[id] += Driv.y / 2.;
				if(ix == Src.x+1 && iy == Src.y+1) dpp[id] += Driv.y / 2.;
				if(ix == Src.x   && iy == Src.y+2) dpp[id] += Driv.y / 2.;
				if(ix == Src.x+1 && iy == Src.y+2) dpp[id] += Driv.y / 2.;

				if(ix == Src.x   && iy == Src.y-1) dpp[id] -= Driv.y / 2.;
				if(ix == Src.x+1 && iy == Src.y-1) dpp[id] -= Driv.y / 2.;
				if(ix == Src.x   && iy == Src.y)   dpp[id] -= Driv.y / 2.;
				if(ix == Src.x+1 && iy == Src.y)   dpp[id] -= Driv.y / 2.;
			}
		}
 	}
}


__global__ void IWBmoving(float* dp, float* dpp, int Nx, int Ny, int2 Src, float Drv, float cfl, int Nyoff, 
		int Nys, int Nye, float b, float2 XY, int iCell, unsigned char* dCell){
	
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;

	float d1, d2, cc;
	float Node[4] = {};
	unsigned char ce;

	cc = cfl;
	d1 = cc * cc * (1 - 2 * b);
	d2 = cc * cc * b;
	ce = 1;
	if(iCell > 0) ce = dCell[id];
	
	if(ce == 1){
		if(ix > 0 && ix < Nx-1 && iy > Nys && iy < Nye){
			dpp[id] = (dp[id+1] + dp[id-1] + dp[id+Nx] + dp[id-Nx] - 4 * dp[id]) * d1
					+ (dp[id+1+Nx] + dp[id+1-Nx] + dp[id-1+Nx] + dp[id-1-Nx] - 4 * dp[id]) * d2
					+ 2 * dp[id] - dpp[id];

			Node[0] = (1 - XY.x) * (1 - XY.y) * Drv / 4;
			Node[1] = (1 + XY.x) * (1 - XY.y) * Drv / 4;
			Node[2] = (1 + XY.x) * (1 + XY.y) * Drv / 4;
			Node[3] = (1 - XY.x) * (1 + XY.y) * Drv / 4;

			if(ix == Src.x   && iy == Src.y  )	dpp[id] += Node[0];
			if(ix == Src.x+1 && iy == Src.y  )	dpp[id] += Node[0];
			if(ix == Src.x   && iy == Src.y+1)	dpp[id] += Node[0];
			if(ix == Src.x+1 && iy == Src.y+1)	dpp[id] += Node[0];

			if(ix == Src.x+1 && iy == Src.y  )	dpp[id] += Node[1];
			if(ix == Src.x+2 && iy == Src.y  )	dpp[id] += Node[1];
			if(ix == Src.x+1 && iy == Src.y+1)	dpp[id] += Node[1];
			if(ix == Src.x+2 && iy == Src.y+1)	dpp[id] += Node[1];

			if(ix == Src.x+1 && iy == Src.y+1)	dpp[id] += Node[2];
			if(ix == Src.x+2 && iy == Src.y+1)	dpp[id] += Node[2];
			if(ix == Src.x+1 && iy == Src.y+2)	dpp[id] += Node[2];
			if(ix == Src.x+2 && iy == Src.y+2)	dpp[id] += Node[2];

			if(ix == Src.x   && iy == Src.y+1)	dpp[id] += Node[3];
			if(ix == Src.x+1 && iy == Src.y+1)	dpp[id] += Node[3];
			if(ix == Src.x   && iy == Src.y+2)	dpp[id] += Node[3];
			if(ix == Src.x+1 && iy == Src.y+2)	dpp[id] += Node[3];
		}
	}
}


__global__ void CE_boundary_Plane2d(float* dp, float* dpp, float ref, unsigned char* dCell, 
	unsigned long long *dBid, unsigned short *dBnode, int Nx, float cfl, int Nbnd){
	
	const unsigned int tx  = threadIdx.x;
	const unsigned int bdx = blockDim.x;
	const unsigned int bx  = blockIdx.x;
	const unsigned int tid = bx * bdx + tx;

	unsigned long long id;
	int n, br, bry, brx;
	unsigned char c;

	if(tid < Nbnd){
		id = dBid[tid];
		if(id > 0){
			n = dBnode[tid];
			c = dCell[id];
			if(n > 0 && c > 1){
				ref = ((1.0 + ref) * cfl - (1.0 - ref)) / ((1.0 + ref) * cfl + (1.0 - ref));
				br  = (int)(n / 9);
				bry = (int)((n - br*9) / 3);
				brx = n % 3;

				if(brx == 1 && bry == 0){
					dp[id] = dpp[id-1] + ref * (dp[id-1] - dpp[id]);
				}
				if(brx == 2 && bry == 0){
					dp[id] = dpp[id+1] + ref * (dp[id+1] - dpp[id]);
				}
				if(brx == 0 && bry == 1){
					dp[id] = dpp[id-Nx] + ref * (dp[id-Nx] - dpp[id]);
				}
				if(brx == 0 && bry == 2){
					dp[id] = dpp[id+Nx] + ref * (dp[id+Nx] - dpp[id]);
				}
			}
		}
	}
	__syncthreads();
}


__global__ void CE_boundary_Edge2d(float* dp, float* dpp, float ref, unsigned char* dCell, 
	unsigned long long *dBid, unsigned short *dBnode, int Nx, float cfl, int Nbnd){
	
	const unsigned int tx  = threadIdx.x;
	const unsigned int bdx = blockDim.x;
	const unsigned int bx  = blockIdx.x;
	const unsigned int tid = bx * bdx + tx;

	unsigned long long id;
	int n, br, bry, brx, jx, jy;
	double ref1, ref2, ref3, px, py, pt;
	double pbx, pby;
	unsigned char c;

	if(tid < Nbnd){
		id = dBid[tid];
		if(id > 0){
			n = dBnode[tid];
			c = dCell[id];
			if(n > 0 && c > 1){
				ref1 = (1.0 - ref) / (sqrt(2.0) * (1.0 + ref) * cfl + (1.0 - ref));
				ref2 = (1.0 + ref) * cfl /sqrt(2.0) / (sqrt(2.0) * (1.0 + ref) * cfl + (1.0 - ref));
				ref3 = ((1.0 + ref) * cfl - (1.0 - ref)) / ((1.0 + ref) * cfl + (1.0 - ref));
				br  = (int)(n / 9);
				bry = (int)((n - br*9) / 3);
				brx = n % 3;

				if(br == 0){
					if(brx == 1 && bry == 1){
						jx = -1;
						jy = -Nx;
						px = dp[id+jy]+dpp[id+jy]+dpp[id]-(dp[id+jx]+dp[id+jx+jy]+dpp[id+jx]+dpp[id+jx+jy]);
						py = dp[id+jx]+dpp[id+jx]+dpp[id]-(dp[id+jy]+dp[id+jx+jy]+dpp[id+jy]+dpp[id+jx+jy]);
						pt = dpp[id]+dpp[id+jx]+dpp[id+jy]+dpp[id+jx+jy]-(dp[id+jx]+dp[id+jy]+dp[id+jx+jy]);
						dp[id] = ref1 * pt - ref2 *(px + py);
					}
					if(brx == 2 && bry == 1){
						jx =  1;
						jy = -Nx;
						px = dp[id+jy]+dpp[id+jy]+dpp[id]-(dp[id+jx]+dp[id+jx+jy]+dpp[id+jx]+dpp[id+jx+jy]);
						py = dp[id+jx]+dpp[id+jx]+dpp[id]-(dp[id+jy]+dp[id+jx+jy]+dpp[id+jy]+dpp[id+jx+jy]);
						pt = dpp[id]+dpp[id+jx]+dpp[id+jy]+dpp[id+jx+jy]-(dp[id+jx]+dp[id+jy]+dp[id+jx+jy]);
						dp[id] = ref1 * pt - ref2 *(px + py);
					}
					if(brx == 1 && bry == 2){
						jx = -1;
						jy =  Nx;
						px = dp[id+jy]+dpp[id+jy]+dpp[id]-(dp[id+jx]+dp[id+jx+jy]+dpp[id+jx]+dpp[id+jx+jy]);
						py = dp[id+jx]+dpp[id+jx]+dpp[id]-(dp[id+jy]+dp[id+jx+jy]+dpp[id+jy]+dpp[id+jx+jy]);
						pt = dpp[id]+dpp[id+jx]+dpp[id+jy]+dpp[id+jx+jy]-(dp[id+jx]+dp[id+jy]+dp[id+jx+jy]);
						dp[id] = ref1 * pt - ref2 *(px + py);
					}
					if(brx == 2 && bry == 2){
						jx = 1;
						jy = Nx;
						px = dp[id+jy]+dpp[id+jy]+dpp[id]-(dp[id+jx]+dp[id+jx+jy]+dpp[id+jx]+dpp[id+jx+jy]);
						py = dp[id+jx]+dpp[id+jx]+dpp[id]-(dp[id+jy]+dp[id+jx+jy]+dpp[id+jy]+dpp[id+jx+jy]);
						pt = dpp[id]+dpp[id+jx]+dpp[id+jy]+dpp[id+jx+jy]-(dp[id+jx]+dp[id+jy]+dp[id+jx+jy]);
						dp[id] = ref1 * pt - ref2 *(px + py);
					}

				}
				else{
					if(brx == 1 && bry == 1){
						pbx = dpp[id-1] + ref3 * (dp[id-1] - dpp[id]);
						pby = dpp[id-Nx] + ref3 * (dp[id-Nx] - dpp[id]);
						dp[id] = (pbx + pby) / 2.0;
					}
					if(brx == 2 && bry == 1){
						pbx = dpp[id+1] + ref3 * (dp[id+1] - dpp[id]);
						pby = dpp[id-Nx] + ref3 * (dp[id-Nx] - dpp[id]);
						dp[id] = (pbx + pby) / 2.0;
					}
					if(brx == 1 && bry == 2){
						pbx = dpp[id-1] + ref3 * (dp[id-1] - dpp[id]);
						pby = dpp[id+Nx] + ref3 * (dp[id+Nx] - dpp[id]);
						dp[id] = (pbx + pby) / 2.0;
					}
					if(brx == 2 && bry == 2){
						pbx = dpp[id+1] + ref3 * (dp[id+1] - dpp[id]);
						pby = dpp[id+Nx] + ref3 * (dp[id+Nx] - dpp[id]);
						dp[id] = (pbx + pby) / 2.0;
					}
				}
			}
		}
	}
 	__syncthreads();
}


__global__ void ABC_Mur(float* dp, float* dpp, int Nx, int Ny, float* dRef, int Nyoff)
{
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;

	if(ix == 0){
		dp[id] = dpp[id+1] + dRef[0] * dp[id+1] - dRef[0] * dpp[id];
	}
	if(ix == Nx-1){
		dp[id] = dpp[id-1] + dRef[1] * dp[id-1] - dRef[1] * dpp[id];
	}
	if(iy == 0){
		dp[id] = dpp[id+Nx] + dRef[2] * dp[id+Nx] - dRef[2] * dpp[id];
	}
	if(iy == Ny-1){
		dp[id] = dpp[id-Nx] + dRef[3] * dp[id-Nx] - dRef[3] * dpp[id];
	}
 	__syncthreads();
}


__global__ void ABC_Higdon_store(float* dpp, float* dp2, int Nx, int Ny, int Nyoff, int Nys, int Nye)
{
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;

	if(iy >= Nys && iy <= Nye){
		if(ix == 0){
			dp2[iy*3] = dpp[id];
			dp2[iy*3+1] = dpp[id+1];
			dp2[iy*3+2] = dpp[id+2];
		}
		if(ix == Nx - 1){
			dp2[iy*3+3*Ny] = dpp[id];
			dp2[iy*3+3*Ny+1] = dpp[id-1];
			dp2[iy*3+3*Ny+2] = dpp[id-2];
		}
	}
	if(iy == 0){
		dp2[ix*3+6*Ny] = dpp[id];
		dp2[ix*3+6*Ny+1] = dpp[id+Nx];
		dp2[ix*3+6*Ny+2] = dpp[id+2*Nx];
	}
	if(iy == Ny - 1){
		dp2[ix*3+6*Ny+3*Nx] = dpp[id];
		dp2[ix*3+6*Ny+3*Nx+1] = dpp[id-Nx];
		dp2[ix*3+6*Ny+3*Nx+2] = dpp[id-2*Nx];
	}
}


__global__ void ABC_Higdon_plane(float* dp, float* dpp, float* dp2, int Nx, int Ny, float* dRef, 
		int Nyoff, int Nys, int Nye, float b1, float b2, float d1, float d2)
{
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;

	if(ix == 0 && iy >= Nys && iy <= Nye){
		if(dRef[0] == 1){
			dp[id] = dp[id+1] + dpp[id+1] - dpp[id];
		}
		else{
			dp[id] = (b1 + b2) * (dp[id+1] - dpp[id])
				   - b1 * b2 * (dp[id+2] - 2.0 * dpp[id+1] + dp2[iy*3])
				   - (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2] - dp2[iy*3+1])
				   + (1 - d1 + 1 - d2) * dpp[id+1]
				   - (1 - d1) * (1 - d2) * dp2[iy*3+2];
		}
	}

	if(ix == Nx - 1 && iy >= Nys && iy <= Nye){
		if(dRef[1] == 1){
			dp[id] = dp[id-1] + dpp[id-1] - dpp[id];
		}
		else{
			dp[id] = (b1 + b2) * (dp[id-1] - dpp[id])
				   - b1 * b2 * (dp[id-2] - 2.0 * dpp[id-1] + dp2[iy*3+3*Ny])
				   - (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2] - dp2[iy*3+3*Ny+1])
				   + (1 - d1 + 1 - d2) * dpp[id-1]
				   - (1 - d1) * (1 - d2) * dp2[iy*3+3*Ny+2];
		}
	}

	if(iy == 0 && ix > 0 && ix < Nx - 1){
		if(dRef[2] == 1){
			dp[id] = dp[id+Nx] + dpp[id+Nx] - dpp[id];
		}
		else{
			dp[id] = (b1 + b2) * (dp[id+Nx] - dpp[id])
				   - b1 * b2 * (dp[id+2*Nx] - 2.0 * dpp[id+Nx] + dp2[ix*3+6*Ny])
				   - (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2*Nx] - dp2[ix*3+6*Ny+1])
				   + (1 - d1 + 1 - d2) * dpp[id+Nx]
				   - (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+2];
		}
	}

	if(iy == Ny - 1 && ix > 0 && ix < Nx - 1){
		if(dRef[3] == 1){
			dp[id] = dp[id-Nx] + dpp[id-Nx] - dpp[id];
		}
		else{
			dp[id] = (b1 + b2) * (dp[id-Nx] - dpp[id])
				   - b1 * b2 * (dp[id-2*Nx] - 2.0 * dpp[id-Nx] + dp2[ix*3+6*Ny+3*Nx])
				   - (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2*Nx] - dp2[ix*3+6*Ny+3*Nx+1])
				   + (1 - d1 + 1 - d2) * dpp[id-Nx]
				   - (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+3*Nx+2];
	}
	}
	__syncthreads();
}


__global__ void ABC_Higdon_corner(float* dp, float* dpp, float* dp2, int Nx, int Ny, float* dRef, 
		int Nyoff, int Nys, int Nye, float b1, float b2, float d1, float d2)
{
	const unsigned int tx  = threadIdx.x;
	const unsigned int ty  = threadIdx.y;
	const unsigned int bdx = blockDim.x;
	const unsigned int bdy = blockDim.y;
	const unsigned int bx  = blockIdx.x;
	const unsigned int by  = blockIdx.y;

	const unsigned int ix = bx * bdx + tx;
	const int iy = by * bdy + ty + Nyoff;
	const int iyd = by * bdy + ty;
	unsigned long long id = Nx * iyd + ix;
	float p1, p2;

	if(ix == 0 && iy == 0){
		p1 = (b1 + b2) * (dp[id+1] - dpp[id])
			- b1 * b2 * (dp[id+2] - 2.0 * dpp[id+1] + dp2[iy*3])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2] - dp2[iy*3+1])
			+ (1 - d1 + 1 - d2) * dpp[id+1]
			- (1 - d1) * (1 - d2) * dp2[iy*3+2];
		p2 = (b1 + b2) * (dp[id+Nx] - dpp[id])
			- b1 * b2 * (dp[id+2*Nx] - 2.0 * dpp[id+Nx] + dp2[ix*3+6*Ny])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2*Nx] - dp2[ix*3+6*Ny+1])
			+ (1 - d1 + 1 - d2) * dpp[id+Nx]
			- (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+2];
		dp[id] = (p1 + p2) / 2.0;
	}
	if(ix == Nx - 1 && iy == 0){
		p1 = (b1 + b2) * (dp[id-1] - dpp[id])
			- b1 * b2 * (dp[id-2] - 2.0 * dpp[id-1] + dp2[iy*3+3*Ny])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2] - dp2[iy*3+3*Ny+1])
			+ (1 - d1 + 1 - d2) * dpp[id-1]
			- (1 - d1) * (1 - d2) * dp2[iy*3+3*Ny+2];
		p2 = (b1 + b2) * (dp[id+Nx] - dpp[id])
			- b1 * b2 * (dp[id+2*Nx] - 2.0 * dpp[id+Nx] + dp2[ix*3+6*Ny])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2*Nx] - dp2[ix*3+6*Ny+1])
			+ (1 - d1 + 1 - d2) * dpp[id+Nx]
			- (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+2];
		dp[id] = (p1 + p2) / 2.0;
	}
	if(ix == 0 && iy == Ny - 1){
		p1 = (b1 + b2) * (dp[id+1] - dpp[id])
			- b1 * b2 * (dp[id+2] - 2.0 * dpp[id+1] + dp2[iy*3])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id+2] - dp2[iy*3+1])
			+ (1 - d1 + 1 - d2) * dpp[id+1]
			- (1 - d1) * (1 - d2) * dp2[iy*3+2];
		p2 = (b1 + b2) * (dp[id-Nx] - dpp[id])
			- b1 * b2 * (dp[id-2*Nx] - 2.0 * dpp[id-Nx] + dp2[ix*3+6*Ny+3*Nx])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2*Nx] - dp2[ix*3+6*Ny+3*Nx+1])
			+ (1 - d1 + 1 - d2) * dpp[id-Nx]
			- (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+3*Nx+2];
		dp[id] = (p1 + p2) / 2.0;
	}
	if(ix == Nx - 1 && iy == Ny - 1){
		p1 = (b1 + b2) * (dp[id-1] - dpp[id])
			- b1 * b2 * (dp[id-2] - 2.0 * dpp[id-1] + dp2[iy*3+3*Ny])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2] - dp2[iy*3+3*Ny+1])
			+ (1 - d1 + 1 - d2) * dpp[id-1]
			- (1 - d1) * (1 - d2) * dp2[iy*3+3*Ny+2];
		p2 = (b1 + b2) * (dp[id-Nx] - dpp[id])
			- b1 * b2 * (dp[id-2*Nx] - 2.0 * dpp[id-Nx] + dp2[ix*3+6*Ny+3*Nx])
			- (b1 * (1-d2) + b2*(1-d1)) * (dpp[id-2*Nx] - dp2[ix*3+6*Ny+3*Nx+1])
			+ (1 - d1 + 1 - d2) * dpp[id-Nx]
			- (1 - d1) * (1 - d2) * dp2[ix*3+6*Ny+3*Nx+2];
		dp[id] = (p1 + p2) / 2.0;
	}
	__syncthreads();
}


void ExcangeBoundary(float* dpt, float* dppt, float* dpb, float* dppb, float* dp, float* dpp, 
	float* mpt, float* mppt, float* mpb, float* mppb, int Nem, int Nbm, int Nydiv, int inode, 
	int gpu_id, int Nygpu)
{
	// ���E�f�[�^�]��(�f�o�C�X���z�X�g)
	if(Nnode > 1 || Ngpu > 1){
		cudaMemcpy(dpt+ 2*Nbm*gpu_id, dp+ Nbm*(Nygpu-Block.y), sizeof(float)*2*Nbm, cudaMemcpyDeviceToHost);
		cudaMemcpy(dppt+2*Nbm*gpu_id, dpp+Nbm*(Nygpu-Block.y), sizeof(float)*2*Nbm, cudaMemcpyDeviceToHost);
		cudaMemcpy(dpb+ 2*Nbm*gpu_id, dp+ Nbm*(Block.y-Boff),  sizeof(float)*2*Nbm, cudaMemcpyDeviceToHost);
		cudaMemcpy(dppb+2*Nbm*gpu_id, dpp+Nbm*(Block.y-Boff),  sizeof(float)*2*Nbm, cudaMemcpyDeviceToHost);
	}

	cudaThreadSynchronize();
	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	// ��w���E�f�[�^�]��(�m�[�h��)
	if(Nnode > 1){
		#pragma omp single
		{
			if(inode == 0){
				MPI_Send((void*)(dpt+ 2*Nbm*(Ngpu-1)), 2*Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, MPI_COMM_WORLD);
				MPI_Send((void*)(dppt+2*Nbm*(Ngpu-1)), 2*Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, MPI_COMM_WORLD);
			}
			else if(inode == Nnode - 1){
				MPI_Recv((void*)mpt,  2*Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
				MPI_Recv((void*)mppt, 2*Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
			}
			else{
				MPI_Sendrecv((void*)(dpt+2*Nbm*(Ngpu-1)), 2*Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, 
							(void*)mpt, 2*Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
				MPI_Sendrecv((void*)(dppt+2*Nbm*(Ngpu-1)), 2*Nbm, MPI_FLOAT, inode+1, HBU_TO_HBD, 
							(void*)mppt, 2*Nbm, MPI_FLOAT, inode-1, HBU_TO_HBD, MPI_COMM_WORLD, &mpi_stat);
			}
		}
	}

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	// ���w���E�f�[�^�]��(�m�[�h��)
	if(Nnode > 1){
		#pragma omp single
		{
			if(inode == 0){
				MPI_Recv((void*)mpb,  2*Nbm, MPI_FLOAT, inode+1, HBD_TO_HBU, MPI_COMM_WORLD, &mpi_stat);
				MPI_Recv((void*)mppb, 2*Nbm, MPI_FLOAT, inode+1, HBD_TO_HBU, MPI_COMM_WORLD, &mpi_stat);
			}
			else if(inode == Nnode - 1){
				MPI_Send((void*)dpb,  2*Nbm, MPI_FLOAT, inode-1, HBD_TO_HBU, MPI_COMM_WORLD);
				MPI_Send((void*)dppb, 2*Nbm, MPI_FLOAT, inode-1, HBD_TO_HBU, MPI_COMM_WORLD);
			}
			else{
				MPI_Sendrecv((void*)dpb, 2*Nbm, MPI_FLOAT, inode-1, HBD_TO_HBU, 
							(void*)mpb, 2*Nbm, MPI_FLOAT, inode+1, HBD_TO_HBU, MPI_COMM_WORLD, &mpi_stat);
				MPI_Sendrecv((void*)dppb, 2*Nbm, MPI_FLOAT, inode-1, HBD_TO_HBU, 
							(void*)mppb, 2*Nbm, MPI_FLOAT, inode+1, HBD_TO_HBU, MPI_COMM_WORLD, &mpi_stat);
			}
		}
	}

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	// ���E�f�[�^�]��(�z�X�g���f�o�C�X)
	if(gpu_id > 0){
		cudaMemcpy(dp,  dpt+ 2*Nbm*(gpu_id-1), sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
		cudaMemcpy(dpp, dppt+2*Nbm*(gpu_id-1), sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
	}
	else{
		if(inode > 0){
			cudaMemcpy(dp,  mpt,  sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
			cudaMemcpy(dpp, mppt, sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
		}
	}
	if(gpu_id < Ngpu - 1){
		cudaMemcpy(dp+ (Nygpu-Boff)*Nbm, dpb+ 2*Nbm*(gpu_id+1), sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
		cudaMemcpy(dpp+(Nygpu-Boff)*Nbm, dppb+2*Nbm*(gpu_id+1), sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
	}
	else{
		if(inode < Nnode-1){
			cudaMemcpy(dp+ (Nygpu-Boff)*Nbm, mpb,  sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
			cudaMemcpy(dpp+(Nygpu-Boff)*Nbm, mppb, sizeof(float)*2*Nbm, cudaMemcpyHostToDevice);
		}
	}
	cudaThreadSynchronize();
	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

}


// �������z�ۑ�
void save_cross_section(float* dp, float* pp, int3 Ndiv, int inode, int gpu_id, int it, int Nydiv, int Boff, 
	int iReg, int ipn, int istx, int isty)
{
	float* pobs  = (float*) malloc(sizeof(float));		 	// �ϑ��_����
	int id, iy, Nd;
	long long idd;
	int ii;
	char filename[200];

	// xy���ʕۑ�
	Nd = Ndx * Ndy;
	idd = 0;

	ii = 0;
	for(int j = 0; j < Nydiv; j++){
		iy = iReg * Nydiv + j;
//		printf("%d %d %d %d\n", Ndx, Ndy, iReg, iy/isty);
		if((iy % isty) == 0){
			for(int i = 0; i < Ndiv.x; i+=istx){
				id = (j + Boff) * Ndiv.x + i;
				cudaMemcpy(pobs, dp+id, sizeof(float), cudaMemcpyDeviceToHost);
				idd = (iy/isty) * Ndx + ii;
//				if(i == 0) printf("%d %d %d %d\n", Ndx, Ndy, iReg, iy/isty);
//				if(idd > Ndx*Ndy) printf("%d %d %d %d\n", Ndx, Ndy, ii, iy/isty);
				pp[idd] = (*pobs);
				ii++;
			}
			ii = 0;
		}
	}

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);
	// �m�[�h0�֓]��
	if(gpu_id == 0 && Nnode > 1){
		for(int ip = 1; ip < Nnode; ip++){
			if(inode == ip)
				MPI_Send((void*)(pp+ip*Nd), Nd, MPI_FLOAT, 0, HBD_TO_HBU, MPI_COMM_WORLD);
			if(inode == 0)
				MPI_Recv((void*)(pp+ip*Nd), Nd, MPI_FLOAT, ip, HBD_TO_HBU, MPI_COMM_WORLD, &mpi_stat);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);
	if(Nnode == 1 || (inode == 0 && gpu_id == 0)){
		if(ipn == 0){
			sprintf(filename, "output%d.dat", it);	// �o�̓t�@�C��
			FILE *fpo  = fopen(filename,"w");
			for(int j = 0; j < Ndy; j++){
				for(int i = 0; i < Ndx; i++){
					id = j * Ndiv.x + i;
					fprintf(fpo, "%e\t", pp[id]);
				}
				fprintf(fpo, "\n");
			}
			fclose(fpo);
		}
		else{
			sprintf(filename, "output%d.bin", it);	// �o�̓t�@�C��
			FILE *fpo  = fopen(filename,"wb");
			fwrite(&Ndx, sizeof(int), 1, fpo);
			fwrite(&Ndy, sizeof(int), 1, fpo);
			fwrite(pp, sizeof(float), Ndx*Ndy, fpo);
			fclose(fpo);
		}
	}
}


void MovingReceiver(float* dp, float* wave, int3 Ndiv, int Nydiv, int3 Block, int iReg, int Boff, Pnt Rcv, float2* RPos, int it){

	float* pobs = (float*) malloc(sizeof(float)*4);		 	// �ϑ��_����
	int ix, iy, itt;
	unsigned long long id;
	float Node[4] = {};
	float pp;
	float2 XY, Prcv, dm;
	int2 Mrcv;
	float rad = angr / 180. * M_PI;

	if(iReceiver == 0) return;
	XY.x = 0;
	XY.y = 0;
	for(int ii = 0; ii < 4; ii++)
		pobs[ii] = 0;
		
	if(iReceiver == 1){
		Prcv.x = RPos[it].x / dl;
		Prcv.y = RPos[it].y / dl;
	}
	if(iReceiver == 2){
		Prcv.x = Rcv.x + (ar * t * t / 2. + vr0 * t) / dl * cos(rad);
		Prcv.y = Rcv.y + (ar * t * t / 2. + vr0 * t) / dl * sin(rad);
	}

	Mrcv.x = int(Prcv.x);
	Mrcv.y = int(Prcv.y);
	if(Mrcv.x > Ndiv.x-1) Mrcv.x = Ndiv.x;
	if(Mrcv.y > Ndiv.y-1) Mrcv.y = Ndiv.y;
	dm.x = Prcv.x - Mrcv.x;
	dm.y = Prcv.y - Mrcv.y;
	XY.x = dm.x * 2 - 1;
	XY.y = dm.y * 2 - 1;
//	printf("%f %f %f %f \n", dm.x, dm.y, XY.x, XY.y);

	if(Mrcv.y / Nydiv == iReg){
		ix = Mrcv.x;
		iy = Mrcv.y % Nydiv;
		id = (iy + Boff) * Ndiv.x + ix;
		cudaMemcpy(pobs,   dp+id, sizeof(float)*2, cudaMemcpyDeviceToHost);
		iy = (Mrcv.y+1) % Nydiv;
		id = (iy + Boff) * Ndiv.x + ix;
		cudaMemcpy(pobs+2, dp+id, sizeof(float)*2, cudaMemcpyDeviceToHost);
	}
	
	Node[0] = (1 - XY.x) * (1 - XY.y) * pobs[0] / 4;
	Node[1] = (1 + XY.x) * (1 - XY.y) * pobs[1] / 4;
	Node[2] = (1 - XY.x) * (1 + XY.y) * pobs[2] / 4;
	Node[3] = (1 + XY.x) * (1 + XY.y) * pobs[3] / 4;
	
	pp = 0;
	for(int i = 0; i < 4; i++)
		pp += Node[i];
	itt = it % Nwave;
	wave[itt] = pp;
//	printf("%f\n", pp);
	if(abs(wave[itt]) > 100.0 || isnan(wave[itt]) != 0){
		printf("Diverged! %d: %e\n", it, wave[itt]);
		#pragma omp barrier
		#pragma omp single
		MPI_Barrier(MPI_COMM_WORLD);
		exit(1);
	}
}

/* ���x�|�e���V����?
__global__ void WaveObss(float* dp, float* dpp, int3 Ndiv, int Nydiv, int3 Block, int iReg, 
		Pnt* dobs, float* dwave, float* du, float cfl, int Nwave, int it, float dt){

	const unsigned int tx  = threadIdx.x;
	const unsigned int bdx = blockDim.x;
	const unsigned int bx  = blockIdx.x;
	const unsigned int io = bx * bdx + tx;

	unsigned long long id;
	float ux, uy, p, phi;

	phi = dobs[io].p / 180. * M_PI;
	if(dobs[io].y / Nydiv == iReg){
		id = (dobs[io].y % Nydiv + Block.y / 2) * Ndiv.x + dobs[io].x;
		if(phi == 0.0){
			dwave[io*Nwave+it] = dp[id];
		}
		else{
			ux = -(dp[id+1] - dp[id-1]) * cfl / 2.;
			uy = -(dp[id+Ndiv.x] - dp[id-Ndiv.x]) * cfl / 2.;
			p  = (dp[id] - dpp[id]);
			dwave[io*Nwave+it] = (ux * cos(phi) + uy * sin(phi) + p) / 2.;
		}
	}
}
*/

__global__ void WaveObss(float* dp, int3 Ndiv, int Nydiv, int3 Block, int iReg, 
		Pnt* dobs, float* dwave, float* du, float cfl, int Nwave, int it){

	const unsigned int tx  = threadIdx.x;
	const unsigned int bdx = blockDim.x;
	const unsigned int bx  = blockIdx.x;
	const unsigned int io = bx * bdx + tx;

	unsigned long long id;
	float ux, uy, phi;

	phi = dobs[io].p / 180. * M_PI;
	if(dobs[io].y / Nydiv == iReg){
		id = (dobs[io].y % Nydiv + Block.y / 2) * Ndiv.x + dobs[io].x;
		if(phi == 0.0){
			dwave[io*Nwave+it] = dp[id];
		}
		else{
			ux = (dp[id+1] - dp[id-1]) * cfl / 2.;
			uy = (dp[id+Ndiv.x] - dp[id-Ndiv.x]) * cfl / 2.;
			du[io] = du[io] + ux * cos(phi) + uy * sin(phi);
			dwave[io*Nwave+it] = (du[io] + dp[id]) / 2.;
//			printf("%f \n", ux * cos(phi) + uy * sin(phi));
		}
	}
}


__global__ void ObssEcho(float* dp, int3 Ndiv, int Nydiv, int3 Block, int iReg, 
		Pnt* dobs, float* dwave, float* dux, float* duy, float cfl, int Nwave, int it){

	const unsigned int tx  = threadIdx.x;
	const unsigned int bdx = blockDim.x;
	const unsigned int bx  = blockIdx.x;
	const unsigned int io = bx * bdx + tx;

	unsigned long long id;

	if(dobs[io].y / Nydiv == iReg){
		id = (dobs[io].y % Nydiv + Block.y / 2) * Ndiv.x + dobs[io].x;
		dux[io] = dux[io] + (dp[id+1] - dp[id-1]) * cfl / 2.;
		duy[io] = duy[io] + (dp[id+Ndiv.x] - dp[id-Ndiv.x]) * cfl / 2.;
		dwave[io*Nwave*3+it] = dp[id];
		dwave[io*Nwave*3+it+Nwave] = dux[io];
		dwave[io*Nwave*3+it+2*Nwave] = duy[io];
//		printf("%f \n", ux * cos(phi) + uy * sin(phi));
	}
}


void SaveWaveBin(float* dwave, float* hwave, float* wave, int it, int gpu_id, int Nydiv, FILE *fpb)
{
//	double ux, uy, phi2, uu;
	int Nw;

	// �ϑ��_�����擾
	Nw = (it % Nwave) + 1;
	cudaMemcpy(hwave, dwave, sizeof(float)*Nwave*Nobs*3, cudaMemcpyDeviceToHost);
		
	for(int io = 0; io < Nobs; io++){
		for(int ii = 0; ii < Nw; ii++){
			if(abs(hwave[io*Nwave*3+ii]) > 1.0e7 || isnan(hwave[io*Nwave*3+ii]) != 0){
				printf("Diverged! %d: %e\n", it, hwave[io*Nwave*3+ii]);
				#pragma omp barrier
				#pragma omp single
				MPI_Barrier(MPI_COMM_WORLD);
				cudaDeviceSynchronize();
				cudaDeviceReset();
				exit(1);
			}
			wave[io*Nwave*3+ii] = hwave[io*Nwave*3+ii];
			wave[io*Nwave*3+Nwave+ii] = hwave[io*Nwave*3+Nwave+ii];
			wave[io*Nwave*3+2*Nwave+ii] = hwave[io*Nwave*3+2*Nwave+ii];
		}
	}

	if(Nnode > 1 && gpu_id == 0){
		for(int io = 0; io < Nobs; io++){
			int inod = obs[io].y / (Ngpu * Nydiv);
			MPI_Bcast((wave+io*Nwave*3), Nwave, MPI_REAL, inod, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	if(Nnode == 1 || (inode == 0 && gpu_id == 0)){
		for(int ii = 0; ii < Nw; ii++){
			for(int io = 0; io < Nobs; io++){
				fwrite(wave+io*Nwave*3+ii, sizeof(float), 1, fpb);
				fwrite(wave+io*Nwave*3+Nwave+ii, sizeof(float), 1, fpb);
				fwrite(wave+io*Nwave*3+2*Nwave+ii, sizeof(float), 1, fpb);
			}
		}
	}
}


void SaveWave(float* dwave, float* hwave, float* wave, int it, int gpu_id, int Nydiv, FILE *fp2)
{
	int Nw;
	
	Nw = (it % Nwave) + 1;
	// �ϑ��_�����擾
	if(iReceiver == 0){
		cudaMemcpy(hwave, dwave, sizeof(float)*Nwave*Nobs*3, cudaMemcpyDeviceToHost);

		for(int io = 0; io < Nobs; io++){
			for(int ii = 0; ii < Nw; ii++){
				if(abs(hwave[io*Nwave*3+ii]) > 1.0e7 || isnan(hwave[io*Nwave*3+ii]) != 0){
					printf("Diverged! %d: %e\n", it, hwave[io*Nwave*3+ii]);
					#pragma omp barrier
					#pragma omp single
					MPI_Barrier(MPI_COMM_WORLD);
					cudaDeviceSynchronize();
					cudaDeviceReset();
					exit(1);
				}
				wave[io*Nwave*3+ii] = hwave[io*Nwave*3+ii];
				wave[io*Nwave*3+Nwave+ii] = hwave[io*Nwave*3+Nwave+ii];
				wave[io*Nwave*3+2*Nwave+ii] = hwave[io*Nwave*3+2*Nwave+ii];
			}
		}
	}
	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	if(Nnode > 1 && gpu_id == 0){
		for(int io = 0; io < Nobs; io++){
			int inod = obs[io].y / (Ngpu * Nydiv);
			MPI_Bcast((wave+io*Nwave*3), Nwave, MPI_REAL, inod, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	#pragma omp barrier
	#pragma omp single
	MPI_Barrier(MPI_COMM_WORLD);

	if(Nnode == 1 || (inode == 0 && gpu_id == 0)){
		for(int ii = 0; ii < Nw; ii++){
			for(int io = 0; io < Nobs; io++){
				fprintf(fp2, "%e,%e,%e,", wave[io*Nwave*3+ii], wave[io*Nwave*3+Nwave+ii],wave[io*Nwave*3+2*Nwave+ii]);
			}
			fprintf(fp2, "\n");
		}
	}
}


