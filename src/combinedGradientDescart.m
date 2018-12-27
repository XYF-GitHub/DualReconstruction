function [g] = combinedGradientDescart(f1,f2,beta1,beta2,a,b,c,d)
%%

%%

[gh1,gl1] = combinedGradient(f1,f2,a,b);
[gh2,gl2] = combinedGradient(f1,f2,c,d);
g = [beta1*gh1(:)+beta2*gh2(:);beta1*gl1(:)+beta2*gl2(:)];

end
