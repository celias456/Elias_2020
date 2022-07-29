function p=invgamma_pdf(x,a,b)

p = ((b^a)/gamma(a))*(x^(-a-1))*exp(-b/x);
% p=b^(-a)*x^(-(a+1))*exp(-1/(b*x))/gamma(a);

if x <= 0
   p = 0;

% if p < 0
%    p = 0;
end
