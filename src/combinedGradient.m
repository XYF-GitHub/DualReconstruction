function [g1,g2] = combinedGradient(f1,f2,a,b)
%%

%%
[row1,col1] = size(f1);
[row2,col2] = size(f2);

ff1 = zeros(row1+4,col1+4);
ff2 = zeros(row2+4,col2+4);

ff1(3:end-2,3:end-2) = f1;
ff2(3:end-2,3:end-2) = f2;

g1 = zeros(size(f1));
g2 = zeros(size(f2));

if row1~=row2 || col1~=col2
    error('Error! Size of f1 and f2 not match!');
end

for i=3:(row1+2)
    for j=3:(col1+2)
        %%%
        v1 = a*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i-1,j)-b*ff2(i-1,j) ) + a*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i,j-1)-b*ff2(i,j-1) );
        v2 = a*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i+1,j)-b*ff2(i+1,j) );
        v3 = a*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i,j+1)-b*ff2(i,j+1) );
        %%%%
        t1 = b*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i-1,j)-b*ff2(i-1,j) ) + b*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i,j-1)-b*ff2(i,j-1) );
        t2 = b*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i+1,j)-b*ff2(i+1,j) );
        t3 = b*( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i,j+1)-b*ff2(i,j+1) );
        %%%
        denom1 = sqrt( eps+( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i-1,j)-b*ff2(i-1,j) )^2+( a*ff1(i,j)+b*ff2(i,j)-a*ff1(i,j-1)-b*ff2(i,j-1) )^2 );
        denom2 = sqrt( eps+( a*ff1(i+1,j)+b*ff2(i+1,j)-a*ff1(i,j)-b*ff2(i,j) )^2+( a*ff1(i+1,j)+b*ff2(i+1,j)-a*ff1(i+1,j-1)-b*ff2(i+1,j-1) )^2 );
        denom3 = sqrt( eps+( a*ff1(i,j+1)+b*ff2(i,j+1)-a*ff1(i-1,j+1)-b*ff2(i-1,j+1) )^2+( a*ff1(i,j+1)+b*ff2(i,j+1)-a*ff1(i,j)-b*ff2(i,j) )^2 );
         
        g1(i-2,j-2) = v1/denom1+v2/denom2+v3/denom3;
        g2(i-2,j-2) = t1/denom1+t2/denom2+t3/denom3;
    end
end













end
