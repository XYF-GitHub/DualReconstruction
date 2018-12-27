close all;
clear all;
clc;

test_data_path = '..\data\testing_set\';
result_path = '..\result\';

N = 256;
prjLen = 1024;

opts.Nx = N;            % Size of the object
opts.Ny = N;            % 
opts.sod = 1000;        % Distance from source to object (mm)
opts.sdd = 1500;        % Distance from source to detector 
opts.dt = 0.388;        % Size of detector voxel 
opts.Uy = prjLen;       % Number of projections
opts.voxel = 1.0;       % opts.voxel = opts.sod/opts.sdd*opts.dt*opts.Uy/opts.Nx;
opts.Nz = 1;            % 
opts.Vz = 1;            %
angcov = 360;
angstp = angcov / (view_num + 1);
theta_vec = 0:angstp:angcov - 1;

systemMatrix_file = '..\data\A_256.mat';
if exist(systemMatrix_file, 'file')
    disp('Loading sysMatrix...');
    load(systemMatrix_file);
else
    tic;
    disp('sysMatrix_2D_all_angle...');
    [W_row, W_col, W_val, sumP2R, sumC, sumR, Row, Col, Val_Num] = sysMatrix_2D_all_angle(theta_vec,opts);
    disp('A sparse...');
    A = sparse(W_row, W_col, W_val, Row, Col);
    clear W_row;
    clear W_col;
    clear W_val;
    toc;
    disp('Saving system matrix...');
    save(systemMatrix_file, '-v7.3', 'A');
end

%%
x1H = 0.0342;
x1L = 0.0588;
x2H = 0.019;
x2L = 0.0251;
a =    x2L/(x1H*x2L - x2H*x1L);
b = -1*x2H/(x1H*x2L - x2H*x1L);
c = -1*x1L/(x1H*x2L - x2H*x1L);
d =    x1H/(x1H*x2L - x2H*x1L);

mat_files = dir(test_data_path);

fopt.angstp = 1; 
fopt.angcov = 360; 
fopt.voxel  = 1.03;  
fopt.filter = 1;      % 1- RL;2- SL;3- cos;4- hamming;5- hann

for f = 3:length(mat_files)
    file = [test_data_path, mat_files(f).name];
    load(file);
    [view_num, prjLen, img_num] = size(mh);
    
    rh = zeros(N, N, img_num);
    rl = zeros(N, N, img_num);
    d1 = zeros(N, N, img_num);
    d2 = zeros(N, N, img_num);
    
    disp(file);
    for n = 1:img_num
        tic;
        disp( sprintf('Image reconstruction %01d / %d ', n, img_num) );
        
        xh = FBP_A(A,double(mh(:,:,n)'),fopt);
        xl = FBP_A(A,double(ml(:,:,n)'),fopt);
        
        rh(:,:,n) = reshape(xh,N,N);
        rl(:,:,n) = reshape(xl,N,N);
        d1(:,:,n) = a*rh(:,:,n) + b*rl(:,:,n);
        d2(:,:,n) = c*rh(:,:,n) + d*rl(:,:,n);
        toc;
    end
    savefileName = [result_path, sprintf('result_test_%04d.mat', f - 2) ];
    disp(['Saving file: ', savefileName]);
    save(savefileName, 'rh', 'rl', 'd1', 'd2');
end

