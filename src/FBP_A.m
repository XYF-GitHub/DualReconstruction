function I = FBP_A(A, p_in, fopt)

% setting
d = 1;                 % Defaults to no cropping of filters frequency response

angstp = fopt.angstp; % É¨Ãè½Ç¼ä¸ô -- ½Ç¶ÈÖÆ
angcov = fopt.angcov; % É¨Ãè½Ç·¶Î§ 180 or 360
voxel  = fopt.voxel;  % ÖØ½¨·Ö±æÂÊ
% ÂË²¨º¯Êý
switch fopt.filter
    case 1 
        filter = 'ram-lak';
    case 2
        filter = 'shepp-logan';
    case 3
         filter = 'cosine';
    case 4
        filter = 'hamming';
    case 5
        filter = 'hann';
    otherwise
        filter = 'ram-lak';
end

% filtering
[p,H] = filterProjections(p_in, filter, d);

coff = (180/angcov) * (angstp/360*pi)/(voxel^2);
proj = p(:)*coff;
I = proj'*A;





function [p,H] = filterProjections(p_in, filter, d)

p = p_in;

% Design the filter
len = size(p,1);
H = designFilter(filter, len, d);

if strcmpi(filter, 'none')
    return;
end

p(length(H),1)=0;  % Zero pad projections

% In the code below, I continuously reuse the array p so as to
% save memory.  This makes it harder to read, but the comments
% explain what is going on.

p = fft(p);    % p holds fft of projections

for i = 1:size(p,2)
    p(:,i) = p(:,i).*H; % frequency domain filtering
end

p = real(ifft(p));     % p is the filtered projections
p(len+1:end,:) = [];   % Truncate the filtered projections

function filt = designFilter(filter, len, d)
% Returns the Fourier Transform of the filter which will be
% used to filter the projections
%
% INPUT ARGS:   filter - either the string specifying the filter
%               len    - the length of the projections
%               d      - the fraction of frequencies below the nyquist
%                        which we want to pass
%
% OUTPUT ARGS:  filt   - the filter to use on the projections


order = max(64,2^nextpow2(2*len*2));

if strcmpi(filter, 'none')
    filt = ones(1, order);
    return;
end

% First create a ramp filter - go up to the next highest
% power of 2.

filt = 2*( 0:(order/2) )./order;
w = 2*pi*(0:size(filt,2)-1)/order;   % frequency axis up to Nyquist

switch filter
    case 'ram-lak'
        % Do nothing
    case 'shepp-logan'
        % be careful not to divide by 0:
        filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
    case 'cosine'
        filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d));
    case 'hamming'
        filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
    case 'hann'
        filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
    otherwise
        eid = sprintf('Images:%s:invalidFilter',mfilename);
        msg = 'Invalid filter selected.';
        error(eid,'%s',msg);
end

filt(w>pi*d) = 0;                      % Crop the frequency response
filt = [filt' ; filt(end-1:-1:2)'];    % Symmetry of the filter