clear;

Filename = 'cell.dat';
%{
Nx = 5000;
Ny = 5000;

Cell = ones(Nx, Ny, 'uint8');
% Cell(round(Nx/3):round(Nx/3)+10, 1:round(Ny/3)) = 2;
% Cell(round(Nx/2):round(Nx/2)+10, round(Ny*2/3):Ny) = 2;

SaveCellData(Filename, Cell, Nx, Ny);
clear Cell Nx Ny;
%}

Cell = ReadCellData(Filename);
Nx = size(Cell, 1);
Ny = size(Cell, 2);
imagesc(Cell');
if find(Cell > 1)
	cmap = [1 1 1
			0 0 0
			0 0 1];
else
	cmap = [1 1 1
			1 1 1];
end
colormap(cmap);
axis xy;
axis image;
hold on;
axis([0 Nx 0 Ny]);
xt = 0 : Nx/5 : Nx;
yt = 0 : Ny/5 : Ny;
xticks(xt);
yticks(yt);
dl = 0.01;
cx = num2cell(xt*dl);
cy = num2cell(yt*dl);
xticklabels(cx);
yticklabels(cy);
hold off;

length(find(Cell==2))
