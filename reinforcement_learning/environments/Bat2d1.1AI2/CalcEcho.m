% 固定送受信点でカージオイドエコーを計算
clear;

cfl = 0.98;				% CFL数
c0 = 340.0;				% 音速
dl = 0.0005;			% グリッド間隔
dt = cfl * dl / c0;		% 時間ステップ
Fs = round(1 / dt);		% サンプリング周波数
Ref = 0.99;				% 障害物の反射率
Freq = 0;				% 0: インパルス，0以外: 放射音波の周波数
Ngpu = 1;				% GPU数

Src = [1000, 1500];		% 入力用音源情報
Range = 3;				% 反射波の観測レンジ(m)
Sep = 0.01;				% 両耳間隔(m)
Angl = 0;				% 左耳角度(deg.)
Angr = 0;				% 右耳角度(deg.)
Angh = 0;				% 頭部角度(deg.)
Angs = 0;				% 超音波放射方向(deg.)
Nt = round(Range * 2 / c0 / dt) + 1;	% 時間ステップ数

p0 = ReadWave(['wave0_x' num2str(Src(1)) '_y' num2str(Src(2))]);
px = ReadWave(['waveX_x' num2str(Src(1)) '_y' num2str(Src(2))]);
py = ReadWave(['waveY_x' num2str(Src(1)) '_y' num2str(Src(2))]);

px = DC_filter(px);
py = DC_filter(py);

ps = (p0 + px * cos(deg2rad(Angs)) + py * sin(deg2rad(Angs)))/2;

p = Receiving_directivity(ps, Angh, Angl, Angr);
p = DC_filter(p);

f = 70000;
t = (0 : 10/(f*dt)) * dt;
s = sin(2*pi*f*t);
pp(:, 1) = fconv(s', p(:, 1));
pp(:, 2) = fconv(s', p(:, 2));
plot(pp);

csvwrite('echo.csv', p);

function p = ReadWave(FileName)
	if isfile([FileName '.bin'])
		fi = fopen([FileName '.bin'],'r');
		Nt = fread(fi, 1, 'int');
		p = transpose(fread(fi, [111 Nt], 'single'));
		fclose(fi);
	elseif isfile([FileName '.csv'])
		p = csvread([FileName '.csv']);
	else
		error('No wave file!');
		quit;
	end
end

function wave = Receiving_directivity(p, Angh, Angl, Angr)

	phil = deg2rad(Angl);
	phir = deg2rad(Angr);

	Nangl = round(Angh / 10) + 9;
	if Nangl > 36
		Nangl = Nangl - 18;
	end
	if Nangl < 0
		Nangl = Nangl + 18;
	end
	Nangr = round(Angh / 10) + 27;
	if Nangr > 36
		Nangr = Nangr - 18;
	end
	if Nangr < 0
		Nangr = Nangr + 18;
	end
	wave(:, 1) = (p(:, Nangl*3+1) + p(:, Nangl*3+2) * cos(phil) + p(:, Nangl*3+3) * sin(phil)) / 2.;
	wave(:, 2) = (p(:, Nangr*3+1) + p(:, Nangr*3+2) * cos(phir) + p(:, Nangr*3+3) * sin(phir)) / 2.;
end


function wave = DC_filter(wave)
	b = ones(1, 5); % 移動平均フィルタ
	for j = 1 : 50
		y = filter(b, 5, wave);
	end
	wave = wave - y;
end

function [y] = fconv(x, h)

	Ly = length(x)+length(h)-1;  % 
	Ly2 = pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly
	X = fft(x, Ly2);		   % Fast Fourier transform
	H = fft(h, Ly2);	           % Fast Fourier transform
	Y = X.*H;        	           % 
	Y(1,:) = 0;
	y = real(ifft(Y, Ly2));      % Inverse fast Fourier transform
	y = y(1:1:Ly);               % Take just the first N elements
%	y=y/max(abs(y));           % Normalize the output
end