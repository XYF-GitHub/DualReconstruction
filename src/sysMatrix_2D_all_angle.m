function [W_row, W_col, W_val, sumP2R,sumC,sumR,Row, Col, Val_Num] = sysMatrix_2D_all_angle(theta_vec,opts)
%% a fast ray-tracing technique for TCT and ECT studies
%% author Cai Ailong
%% from VCT Lab
%% date 2012-12-15
%% function Description: this short code deals with the weighted matrix in
%% conebeam CT by circular line 
%% parameters :
%% theta_i   rotation angel, it is a scalar
%% v_id      the specific line number of the flat panel detector, it is a
%% scaler
%% opts   it is a structor where:
%% opts.Nx, opts.Ny, opts.Nz: the scale of object
%% opts.voxel: the voxel size of the object
%% opts.dt: the pixel size of the detector
%% opts.sod source to object distance
%% opts.sdd source to detector distance
%% opts.Uy number of columns of pixels on the detector
%% opts.Vz number of rows of pixels on the detector
%%      Siddon's method: for a Nx*Ny*Nz voxel array, the idea of siddon's method is the
%% consideratin of the voxels on a ray as the intersection of that ray
%% with orthogonal sets of equally spaced, parrallel planes. For detailed
%% information, please refer to the paper 'A Fast Ray-Tracing for TCT and
%% ECT Studies ', author by Guoping Han, Zhengrong Liang(������), and Jiangsheng
%% You���Ƚ�����
%% correct a mistake at 2013-01-23 0:45
%% User should pay attention to the rotation direction: clockwise or anticlockwise?
%% �����ģ
% opts.Nx = 128;
% opts.Ny = 128;
% opts.Nz = 128;
% opts.voxel = 0.5;
% opts.dt = 0.25;
% opts.sdd = 800;
% opts.sod = 1600;
% opts.Uy = 256;
% opts.Vz = 256;
% figure;
Nx = opts.Nx;
Ny = opts.Ny;
%% ̽������������
voxel = opts.voxel;%% �����ݵ�С����ı߳�
% dt = opts.dt;%%̽������̽Ԫ�ߴ�
ratio = opts.dt/opts.voxel;%% �����ر߳���Ϊ��һ����׼������̽�����ķֱ泤��(�������)
sdd = opts.sdd/opts.voxel;
sod = opts.sod/opts.voxel;
%% ̽����̽Ԫ��Ŀ �߶�row = Vz�����col = U����row*col = Vz*Uy��
Uy = opts.Uy;%% ̽����̽Ԫ����
%% ͶӰ������ϡ�������ʾ
N_ang = length(theta_vec);
rayNum = N_ang*Uy;
volNum = Nx*Ny;
% W = zeros(rayNum,volNum);
%% estimate the number of non-entries
Num_NonEty = floor(rayNum*sqrt(Nx^2 + Ny^2)/1.4);
%%
W_row = zeros(Num_NonEty,1);
W_col = zeros(Num_NonEty,1);
W_val = zeros(Num_NonEty,1);
Row = rayNum;
Col = volNum;
sumC = zeros(Col,1);
sumR = zeros(Row,1);
%%
sumP2R = zeros(Row,1);
%%
% Wt_row = zeros(Num_NonEty,1);
% Wt_col = zeros(Num_NonEty,1);
% Wt_val = zeros(Num_NonEty,1);
% Rowt = volNum;
% Colt = rayNum;
%%
Val_Num = 1;
%% ��ת���ĵ�����
%% 
O_x = Nx/2;
O_y = Ny/2;
for ang = 0:(N_ang-1)
    theta_i = theta_vec(ang+1);
    disp(theta_i);
%% ̽����̽Ԫ������
for u_id = 0:1:(Uy-1)
gamma = atan(sdd/((u_id - Uy/2 + 0.5)*ratio));
amp = 1/sin(abs(gamma));
%% �����������
%% ��ת�Ƕ�
theta = -1*theta_i*pi/180;
%% ̽�������ĵ�����(δ��ת)
Dx = sdd - sod + O_x;
Dy = O_y;
%% ��̽Ԫ���ĵ���ά����(δ��ת)
u_y_ur = Dy + (u_id + 0.5 - Uy/2)*ratio; %%
w_x_ur = Dx;
%% ��ת���̽Ԫ����
w_x = (w_x_ur - O_x)*cos(theta) - (u_y_ur - O_y)*sin(theta) + O_x;
u_y = (w_x_ur - O_x)*sin(theta) + (u_y_ur - O_y)*cos(theta) + O_y;

%% (��ת��)��Դ����
S_x = (-sod)*cos(theta) - (0)*sin(theta) + O_x;
S_y = (-sod)*sin(theta) + (0)*cos(theta) + O_y;

%% I  ��ʼ�������������6������ռ�ֱ�ߵĽ��㣬��ȷ������ͳ����
%% 1.���������6������ռ�ֱ�ߵĽ���
% x \in 0 and Nx: lamda_x0 lamda_Nx lamda_x = (x_pzm - S_x)/(w_x - S_x)
% y \in 0 and Ny: lamda_y0 lamda_Ny lamda_y = (y_pzm - S_y)/(u_y - S_y)
%%  x 0 Nx
lamda_x0 = (0 - S_x)/(w_x - S_x);
x_pzm_x0 = S_x + lamda_x0*(w_x - S_x); %P1 temp
y_pzm_x0 = S_y + lamda_x0*(u_y - S_y); %P1 temp

lamda_Nx = (Nx - S_x)/(w_x - S_x);
x_pzm_Nx = S_x + lamda_Nx*(w_x - S_x); %P2 temp
y_pzm_Nx = S_y + lamda_Nx*(u_y - S_y); %P2 temp
%% y 0 Ny
lamda_y0 = (0 - S_y)/(u_y - S_y);
x_pzm_y0 = S_x + lamda_y0*(w_x - S_x); %P3 temp
y_pzm_y0 = S_y + lamda_y0*(u_y - S_y); %P3 temp

lamda_Ny = (Ny - S_y)/(u_y - S_y);
x_pzm_Ny = S_x + lamda_Ny*(w_x - S_x); %P4 temp
y_pzm_Ny = S_y + lamda_Ny*(u_y - S_y); %P4 temp

%% 2.Ѱ�������ͳ����
X = [x_pzm_x0 x_pzm_Nx x_pzm_y0 x_pzm_Ny];% x_pzm_z0 x_pzm_Nz];
Y = [y_pzm_x0 y_pzm_Nx y_pzm_y0 y_pzm_Ny];% y_pzm_z0 y_pzm_Nz];

eps = 1E-10;
% ind_X = find( (X >= (0-eps)) & (X <= (Nx+eps)) );
% ind_Y = find( (Y >= (0-eps)) & (Y <= (Ny+eps)) );

ind_XY = find((X >= (0-eps)) & (X <= (Nx+eps))& (Y >= (0-eps)) & ...
    (Y <= (Ny+eps)));

    if (~isempty(ind_XY))
        x_start_tmp = X(ind_XY(1));% start ->1
        y_start_tmp = Y(ind_XY(1));

        x_end_tmp = X(ind_XY(2));% end ->2
        y_end_tmp = Y(ind_XY(2));

        L = sqrt((x_start_tmp-x_end_tmp)^2 + (y_start_tmp-y_end_tmp)^2);
%% II siddon ray tracing ���֣������������򡢼��㽻��������Ȩֵ
%% 1.�������ͳ����Ϊ�����յ�Կռ�ֱ����������
        plane_x = 0:1:Nx;
        plane_y = 0:1:Ny;
        alpha_x = (plane_x - x_start_tmp)/(x_end_tmp - x_start_tmp);
        alpha_y = (plane_y - y_start_tmp)/(y_end_tmp - y_start_tmp);
%% 2.�����ںϼ����� 
        alpha_xy = [alpha_x alpha_y];
%         ind_alpha_xy = find((alpha_xy >=(0 - eps)) & (alpha_xy <= (1+eps)) );
        alpha_xy_pick = alpha_xy((alpha_xy >=(0 - eps)) & (alpha_xy <= (1+eps)) );
        val = sort(alpha_xy_pick , 'ascend');
        m = length(val);
%% 3.����׷����ѭ�������㽻��������Ȩֵ
        for i = 1:(m-1)
            l = (val(i+1) - val(i))*L;
            alpha_mid = 1/2*(val(i+1)+val(i));
            ind_x = floor(x_start_tmp + alpha_mid*(x_end_tmp - x_start_tmp));
            ind_y = floor(y_start_tmp + alpha_mid*(y_end_tmp - y_start_tmp));
            ii = ang*Uy + u_id +1;
            jj = ind_y*Nx + ind_x +1; 
%            if()
                if (jj >= 1 && jj <= Nx*Ny)
%                     disp(ii);
%                   W(ii,jj) = l*amp;    
                    W_row(Val_Num) = ii;
                    W_col(Val_Num) = jj;
                    W_val(Val_Num) = l*amp*voxel;
                    Val_Num = Val_Num + 1;
                    sumC(jj) = sumC(jj) + l*amp*voxel;
                    sumR(ii) = sumR(ii) + l*amp*voxel;
                    sumP2R(ii) = sumP2R(ii) + l*amp*voxel*l*amp*voxel;
                end     
        end
    end %% 
end%%
Val_Num = Val_Num - 1;
W_row = W_row(W_row ~= 0);
W_col = W_col(W_col ~= 0);
W_val = W_val(W_col ~= 0);%% ��W_col��W_val
end
    
    
    
    
    
    
    
    
    