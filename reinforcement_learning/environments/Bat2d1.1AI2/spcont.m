% ‰¹ˆ³•ª•z}•`‰æŠÖ” 2018.2.15
function spcont(it)
	filename = ['output' num2str(it) '.bin'];
	a = exist(filename, 'file');
	if a == 0
		error('No output*** file!');
	end
	fi = fopen(['output' num2str(it) '.bin'], 'r');
	m = fread(fi, 1, 'int32');
	n = fread(fi, 1, 'int32');
	p = transpose(fread(fi, [m n], 'single'));
	fclose(fi);
	imagesc(p);
	
	colormap jet;
	colorbar;
	axis equal; axis xy;
	axis([0 size(p, 2) 0 size(p, 1)]);
%	saveas(gcf, ['fig' num2str(n) '.png']);
%	savefig(gcf, ['fig' num2str(it)], 'compact');
end
