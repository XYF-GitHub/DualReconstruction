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
opts.voxel = 1.03;      % opts.voxel = opts.sod/opts.sdd*opts.dt*opts.Uy/opts.Nx;
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

Niter = 100;
x1H = 0.0342;
x1L = 0.0588;
x2H = 0.019;
x2L = 0.0251;
a =    x2L/(x1H*x2L - x2H*x1L);
b = -1*x2H/(x1H*x2L - x2H*x1L);
c = -1*x1L/(x1H*x2L - x2H*x1L);
d =    x1H/(x1H*x2L - x2H*x1L);

mat_files = dir(test_data_path);

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
        tic
        disp( sprintf('Image reconstruction %01d / %d ', n, img_num) );
        ml_slice = ml(:,:,n)';
        mh_slice = mh(:,:,n)';
        ml_slice = double(ml_slice(:));
        mh_slice = double(mh_slice(:));
        
        xh = zeros(N*N,1);
        xl = zeros(N*N,1);
        xh_old = zeros(N*N,1);
        xl_old = zeros(N*N,1);
        x = zeros(2*N*N,1);
        px = zeros(2*N*N,1);
        px_old = zeros(2*N*N,1);

        beta1 = 0.003;
        beta2 = 0.0045;
        beta1_red = 0.99;
        beta2_red = 0.99;
        
        kai = 0.3;
        tol = eps;
        
        for i = 1:Niter
            x = [xh;xl];
            gtv = combinedGradientDescart(reshape(xh,N,N),reshape(xl,N,N),beta1,beta2,a,b,c,d);
            gquadh = A'*(A*double(xh) - mh_slice);
            gquadl = A'*(A*double(xl) - ml_slice);
            g = single(gtv + [gquadh(:);gquadl(:)]);
    
            px(:) = 0;
            px(g <= 0 | x > 0) = g(g <= 0 | x > 0);
            if i == 1
                xh_old = single(zeros(N*N,1));
                xl_old = single(zeros(N*N,1));
                x_old = [xh_old;xl_old];
                px_old = single(zeros(2*N*N,1));
                alpha = 10E-8;
            else
                alpha1 = (x - x_old)'*(x - x_old) / ((x - x_old)'*(px - px_old));
                alpha2 = (x - x_old)'*(px - px_old) / ((px - px_old)'*(px - px_old));
                if alpha2 / alpha1 < kai
                    alpha = alpha1;
                else
                    alpha = alpha2;
                end
                x_old = x;
                px_old = px;
            end
            x = x - alpha*px;
            x(x < 0) = 0;
            xh = x(1:N*N);
            xl = x((N*N + 1):end);
            
            if norm(x - x_old) <= tol
                break;
            end
            beta1 = beta1*beta1_red;
            beta2 = beta2*beta2_red;            
        end
        
        rh(:,:,n) = reshape(xh,N,N);
        rl(:,:,n) = reshape(xl,N,N);
        d1(:,:,n) = a*rh(:,:,n) + b*rl(:,:,n);
        d2(:,:,n) = c*rh(:,:,n) + d*rl(:,:,n);
%         figure(1); imshow(rh(:,:,n),[]), title(['rh image: ', num2str(n)]);
%         figure(2); imshow(rl(:,:,n),[]), title(['rl image: ', num2str(n)]);
%         figure(3); imshow(d1(:,:,n),[]), title(['d1 image: ', num2str(n)]);
%         figure(4); imshow(d2(:,:,n),[]), title(['d2 image: ', num2str(n)]);
        toc;
    end
    savefileName = [result_path, sprintf('result_test_%04d.mat', f - 2) ];
    disp(['Saving file: ', savefileName]);
    save(savefileName, 'rh', 'rl', 'd1', 'd2');
end

