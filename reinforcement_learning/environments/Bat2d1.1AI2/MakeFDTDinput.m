% MATLAB Linux実行用ファイル
% 固定送受信点でエコーを計算
clear;
clear global;

global cfl c0 dl dt Size Src Rcv Nt Ref Freq Ngpu iCell

iCell = 1;				% 0: 矩形領域, 1: 任意領域(障害物)

cfl = 0.98;				% CFL数
c0 = 340.0;				% 音速
dl = 0.0005;			% グリッド間隔
dt = cfl * dl / c0;		% 時間ステップ
Fs = round(1 / dt);		% サンプリング周波数
Ref = 0.99;				% 障害物の反射率
Freq = 0;				% 0: インパルス，0以外: 放射音波の周波数
Ngpu = 1;				% GPU数

Size = [3.5, 1.5];		% 領域サイズ(m)
Ndiv = round(Size / dl);	% 分割数

S = [0.9, 0.75];		% 音源位置(m)
IS = round(S / dl);		% 音源位置(デルタ)
Src = [IS, 0];			% 入力用音源情報
Range = 3;				% 反射波の観測レンジ(m)
Sep = 0.01;				% 両耳間隔(m)
Nt = round(Range * 2 / c0 / dt) + 1;	% 時間ステップ数

% 受音点設定
Rcv = [IS,  0];

% 領域設定
if iCell > 0
	Cell = MakeCell;
%	length(find(Cell>1))
%	DisplayCell(Cell, dl, 1);
end

% 入力データ保存
Filename = 'cell.dat';
SaveCellData(Filename, Cell, Ndiv(1), Ndiv(2));
MakeInput;

function Cell = MakeCell
global cfl c0 dl dt Size Src Rcv Nt Ref Freq Ngpu iCell

	Ndiv = round(Size / dl);
	Nx = Ndiv(1);
	Ny = Ndiv(2);

	Cell = ones(Nx, Ny, 'uint8');
%	Cell(Nx, :) = 0;	% 奥の壁
%	Cell(:, 1) = 0;		% 床
%	Cell(:, Ny) = 0;	% 天井

% 第1アクリル
	Nx1 = round(1 / dl);
	Ny1 = Ny - round(0.66 / dl);
	Cell(Nx1-2, Ny1:Ny) = 2;
	Cell(Nx1+2, Ny1:Ny) = 2;
	Cell(Nx1-2:Nx1+2, Ny1) = 2;
	Cell(Nx1-1:Nx1+1, Ny1+1:Ny) = 0;

% 第2アクリル
	Nx2 = round(2 / dl);
	Ny2 = round(0.66 / dl);
	Cell(Nx2-2, 1:Ny2) = 2;
	Cell(Nx2+2, 1:Ny2) = 2;
	Cell(Nx2-2:Nx2+2, Ny2) = 2;
	Cell(Nx2-1:Nx2+1, 1:Ny2-1) = 0;

% 第3アクリル	
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
	In{1} = '//2次元IWB-FDTD法　移動音源/受音点用(自動生成)';
	In{2}{1} = num2str(iCell);
	In{2}{2} = num2str(1);
	In{2}{3} = num2str(1);
	In{2}{4} = ' // モデルタイプ(0:矩形,1:任意), 手法 (0:SLF, 1:IWB), 境界条件(0:Mur1次,1:Higdon2次)';
	In{3}{1} = num2str(Ndiv(1));
	In{3}{2} = num2str(Ndiv(2));
	In{3}{3} = 'cell.dat';
	In{3}{4} = '	// x, y方向分割数Nx, Ny, セルデータ名(どちらか指定)';
	In{4}{1} = '0.0 0.0 0.0 0.0';
	In{4}{2} = num2str(Ref);
	In{4}{3} = '// -x,x,-y,y境界条件, 任意境界反射率';
	In{5}{1} = num2str(cfl);
	In{5}{2} = num2str(dl);
	In{5}{3} = num2str(c0);
	In{5}{4} = '	// CFL, dl, c0';
	In{6}{1} = num2str(Nt);
	In{6}{2} = '			// 計算ステップ数Nt';
	In{7}{1} = num2str(0);
	In{7}{2} = 'none	// 音源移動方法(0:固定，1:座標ファイル，2:直線移動)';
	In{8}{1} = num2str(Src(1));
	In{8}{2} = num2str(Src(2));
	In{8}{3} = num2str(0);
	In{8}{4} = num2str(0);
	In{8}{5} = num2str(0);
	In{8}{6} = num2str(0);
	In{8}{7} = '// (ファイルの時は無視)音源位置src_x, y, 方向(φ), 音源速度, 加速度，角度';
	In{9}{1} = num2str(Freq);
	In{9}{2} = '10';
	In{9}{3} = 'none';
	In{9}{4} = '	// freq(-1:ファイル，0:インパルス), バースト波数Nd (0: 連続, 1:微分ガウス)';
	In{10}{1} = num2str(0);
	In{10}{2} = 'none	// 受音点移動方法(0:固定，1:座標ファイル，2:直線移動)';
	In{11} = '0 1 				//音圧分布出力(0:なし), 波形';
	In{12} = '1000 4 4 0 300000 		// 時間間隔，音圧分布間引き(x,y方向)，開始，終了';
	In{13}{1} = num2str(Ngpu);
	In{13}{2} = ' 1				// GPU数/ノード, GPU ID';

	Fo = fopen('input.dat', 'w', 'n', 'Shift_JIS');
	for li = 1 : length(In)
		Line = string(join(In{li},'  '));
		fprintf(Fo, string(Line));
		fprintf(Fo, newline);
	end
	fclose('all');
end
