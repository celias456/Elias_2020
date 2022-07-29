function [ml] = mlike(store_theta,prior,like,alp)
%Calculates the log marginal likelihood

[nsim,m] = size(store_theta);
theta_hat = mean(store_theta)';
Qtheta = cov(store_theta);
chi2q = chi2inv(1-alp,m);
Qinv = Qtheta^-1;
determ = det(Qtheta);
const_f = log(1/alp) - m/2*log(2*pi) - 0.5*log(determ);
%const_f = real(const_f);
test = zeros(nsim,1);
f = zeros(nsim,1);
store_w = -inf(nsim,1);


for i = 1:nsim
    theta = store_theta(i,:)';
    l = like(i);
    pr = prior(i); 
    dev = theta-theta_hat;
    test(i) = dev'*Qinv*dev;
    if test(i) <= chi2q
       f(i) = const_f - 0.5*test(i);
       store_w(i) = f(i) - (pr + l);
    end
end
maxllike = max(store_w);
num1 = store_w - maxllike;
num2 = exp(num1);
num3 = mean(num2);
num4 = log(num3);
num5 = num4 + maxllike;
ml = -num5;
% ml = log(mean(exp(store_w-maxllike))) + maxllike;
% ml = -ml;

end

