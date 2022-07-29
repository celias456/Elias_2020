function [p_Y] = modified_harmonic_mean(theta,log_prior,log_likelihood,tau)
%Calculates the log marginal likelihood using the modified harmonic mean

%This code and notation comes from "Bayesian Estimation of DSGE Models" by
%Herbst and Schorfheide, pages 94-95

% Inputs:
% theta: N x d matrix of values of parameter draws calculated from a metropolis-hastings algorithm
% log_prior: vector of values of the log prior associated with the parameter draws in the matrix "theta"
% log_likelihood: vector of values of the log likelihood associated with the parameter draws in the matrix "theta"
% tau: tuning parameter (should be a number close to 1)

% Output:
% ml: value of the marginal likelihood

%N is the number of simulations, d is the number of parameters in theta
[N,d] = size(theta);

%Calculate the posterior mean
theta_bar = mean(theta)';

%Calculate the posterior covariance matrix
V_bar_theta = cov(theta);

%Find the determinate of the posterior covariance matrix
V_bar_theta_determinate = det(V_bar_theta);

%Calculate the inverse of the posterior covariance matrix
if V_bar_theta_determinate == 0 %Posterior covariance matrix is singular; Take the p-inverse
    V_bar_theta_inv = pinv(V_bar_theta);
else
    V_bar_theta_inv  = V_bar_theta^-1;
end

%Calculate the inverse cdf of a chi-squared distribution with d degrees of
%freedom
F_chi_square_inv = chi2inv(tau,d);

%Calculate the normalizing constant for f
const_f = log(1/tau) - (d/2)*log(2*pi) - 0.5*log(V_bar_theta_determinate);
const_f = real(const_f); %Ensures that the normalizing constant doesn't include any imaginary parts

%Create storage
test = zeros(N,1);
f = zeros(N,1);
p_Y_elements = -inf(N,1); %Initial values are negative infinity in logs (0 in levels)

%Begin the summation
for i = 1:N 
    theta_i = theta(i,:)';
    log_likelihood_i = log_likelihood(i);
    log_prior_i = log_prior(i);
    deviation = theta_i-theta_bar;
    test(i) = deviation'*V_bar_theta_inv*deviation;
    if test(i) < F_chi_square_inv
       f(i) = const_f - 0.5*test(i);
       p_Y_elements(i) = f(i) - (log_likelihood_i + log_prior_i);
    end


%The average value of the values in "p_Y_elements" must be found; The values are
%in logarithms and must be "transformed". See the Excel spreadsheet
%"Modified Harmonic Mean" example to see the transformation at work.

maxllike = max(p_Y_elements);
num1 = p_Y_elements - maxllike;
num2 = exp(num1);
num3 = mean(num2);
num4 = log(num3);
num5 = num4 + maxllike;
p_Y = -num5;
end



end

