% MATLAB Linux���s�p�t�@�C��
% �Œ著��M�_�ŃG�R�[���v�Z
clear;
clear global;

global cfl c0 dl dt Size Src Rcv Nt Ref Freq Ngpu iCell

iCell = 1;				% 0: ��`�̈�, 1: �C�ӗ̈�(��Q��)

cfl = 0.98;				% CFL��
c0 = 340.0;				% ����
dl = 0.0005;			% �O���b�h�Ԋu
dt = cfl * dl / c0;		% ���ԃX�e�b�v
Fs = round(1 / dt);		% �T���v�����O���g��
Ref = 0.99;				% ��Q���̔��˗�
Freq = 0;				% 0: �C���p���X�C0�ȊO: ���ˉ��g�̎��g��
Ngpu = 1;				% GPU��

Size = [3.5, 1.5];		% �̈�T�C�Y(m)
Ndiv = round(Size / dl);	% ������

S = [0.9, 0.75];		% �����ʒu(m)
IS = round(S / dl);		% �����ʒu(�f���^)
Src = [IS, 0];			% ���͗p�������
Range = 3;				% ���˔g�̊ϑ������W(m)
Sep = 0.01;				% �����Ԋu(m)
Nt = round(Range * 2 / c0 / dt) + 1;	% ���ԃX�e�b�v��

% �󉹓_�ݒ�
Rcv = [IS,  0];

% �̈�ݒ�
if iCell > 0
	Cell = MakeCell;
%	length(find(Cell>1))
%	DisplayCell(Cell, dl, 1);
end

% ���̓f�[�^�ۑ�
Filename = 'cell.dat';
SaveCellData(Filename, Cell, Ndiv(1), Ndiv(2));
MakeInput;

function Cell = MakeCell
global cfl c0 dl dt Size Src Rcv Nt Ref Freq Ngpu iCell

	Ndiv = round(Size / dl);
	Nx = Ndiv(1);
	Ny = Ndiv(2);

	Cell = ones(Nx, Ny, 'uint8');
%	Cell(Nx, :) = 0;	% ���̕�
%	Cell(:, 1) = 0;		% ��
%	Cell(:, Ny) = 0;	% �V��

% ��1�A�N����
	Nx1 = round(1 / dl);
	Ny1 = Ny - round(0.66 / dl);
	Cell(Nx1-2, Ny1:Ny) = 2;
	Cell(Nx1+2, Ny1:Ny) = 2;
	Cell(Nx1-2:Nx1+2, Ny1) = 2;
	Cell(Nx1-1:Nx1+1, Ny1+1:Ny) = 0;

% ��2�A�N����
	Nx2 = round(2 / dl);
	Ny2 = round(0.66 / dl);
	Cell(Nx2-2, 1:Ny2) = 2;
	Cell(Nx2+2, 1:Ny2) = 2;
	Cell(Nx2-2:Nx2+2, Ny2) = 2;
	Cell(Nx2-1:Nx2+1, 1:Ny2-1) = 0;

% ��3�A�N����	
	Nx3 = round(3 / dl);
	Ny3 = Ny - round(0.66 / dl);
 	Cell(Nx3-2, Ny3:Ny) = 2;
 	Cell(Nx3+2, Ny3:Ny) = 2;
 	Cell(Nx3-2:Nx3+2, Ny3) = 2;
 	Cell(Nx3-1:Nx3+1, Ny3+1:Ny) = 0;

end

function DisplayCell(Cell, dl, sep)
	imagesc(Cell');
	if find(Cell > 1)
		cmap = [0 0 0
				1 1 1
				0 0 1];
	else
		cmap = [1 1 1
				1 1 1
				0 0 1];
	end
	colormap(cmap);
	axis xy;
	axis image;
	Nx = size(Cell, 1);
	Ny = size(Cell, 2);
	axis([0 Nx 0 Ny]);
% 	isep = round(sep / dl);
% 	xt = 0:isep:Nx;
% 	yt = 0:isep:Ny;
% 	xticks(xt);
% 	yticks(yt);
% 	cx = num2cell(xt*dl);
% 	cy = num2cell(yt*dl);
% 	xticklabels(cx);
% 	yticklabels(cy);
end


function MakeInput
global cfl c0 dl dt Size Src Rcv Nt Ref Freq Ngpu iCell

	Ndiv = round(Size / dl);
	In{1} = '//2����IWB-FDTD�@�@�ړ�����/�󉹓_�p(��������)';
	In{2}{1} = num2str(iCell);
	In{2}{2} = num2str(1);
	In{2}{3} = num2str(1);
	In{2}{4} = ' // ���f���^�C�v(0:��`,1:�C��), ��@ (0:SLF, 1:IWB), ���E����(0:Mur1��,1:Higdon2��)';
	In{3}{1} = num2str(Ndiv(1));
	In{3}{2} = num2str(Ndiv(2));
	In{3}{3} = 'cell.dat';
	In{3}{4} = '	// x, y����������Nx, Ny, �Z���f�[�^��(�ǂ��炩�w��)';
	In{4}{1} = '0.0 0.0 0.0 0.0';
	In{4}{2} = num2str(Ref);
	In{4}{3} = '// -x,x,-y,y���E����, �C�Ӌ��E���˗�';
	In{5}{1} = num2str(cfl);
	In{5}{2} = num2str(dl);
	In{5}{3} = num2str(c0);
	In{5}{4} = '	// CFL, dl, c0';
	In{6}{1} = num2str(Nt);
	In{6}{2} = '			// �v�Z�X�e�b�v��Nt';
	In{7}{1} = num2str(0);
	In{7}{2} = 'none	// �����ړ����@(0:�Œ�C1:���W�t�@�C���C2:�����ړ�)';
	In{8}{1} = num2str(Src(1));
	In{8}{2} = num2str(Src(2));
	In{8}{3} = num2str(0);
	In{8}{4} = num2str(0);
	In{8}{5} = num2str(0);
	In{8}{6} = num2str(0);
	In{8}{7} = '// (�t�@�C���̎��͖���)�����ʒusrc_x, y, ����(��), �������x, �����x�C�p�x';
	In{9}{1} = num2str(Freq);
	In{9}{2} = '10';
	In{9}{3} = 'none';
	In{9}{4} = '	// freq(-1:�t�@�C���C0:�C���p���X), �o�[�X�g�g��Nd (0: �A��, 1:�����K�E�X)';
	In{10}{1} = num2str(0);
	In{10}{2} = 'none	// �󉹓_�ړ����@(0:�Œ�C1:���W�t�@�C���C2:�����ړ�)';
	In{11} = '0 1 				//�������z�o��(0:�Ȃ�), �g�`';
	In{12} = '1000 4 4 0 300000 		// ���ԊԊu�C�������z�Ԉ���(x,y����)�C�J�n�C�I��';
	In{13}{1} = num2str(Ngpu);
	In{13}{2} = ' 1				// GPU��/�m�[�h, GPU ID';

	Fo = fopen('input.dat', 'w', 'n', 'Shift_JIS');
	for li = 1 : length(In)
		Line = string(join(In{li},'  '));
		fprintf(Fo, string(Line));
		fprintf(Fo, newline);
	end
	fclose('all');
end
