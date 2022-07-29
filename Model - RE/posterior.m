function [post] = posterior(parameters)
%Calculates the value of the posterior

%% Characteristics of model
variables_end = 5; %Number of endogenous variables
variables_exo = 3; %Number of exogenous variables
variables_exp = 2; %Number of expectational variables
num_parameters = 10; %Number of parameters to be estimated
variables_state = 2; %Number of state variables

%% Data 

data = csvread('Data.csv'); 
H_re = [1,0,0,0,0,0,0
        0,1,0,0,0,0,0;
        0,0,1,0,0,0,0];
    
T = size(data,1);
    
%% Initial values for KF
k1 = 1; %Mean of prior distribution of state
k2 = 3; %Variance of prior distribution of state
    
%% Values of parameters

sigma = parameters(1);
beta = parameters(2);
kappa = parameters(3);
r_pi = parameters(4);
r_x = parameters(5);
sigma_x = parameters(6);
sigma_pi = parameters(7);
sigma_i = parameters(8);
rho_x = parameters(9);
rho_pi = parameters(10);

%% Priors
prior_sigma = [16;0.125]; %Gamma pdf; First element is k (shape) parameter, second element is theta (scale) parameter
prior_beta = [99;1]; %Beta pdf; First element is alpha, second element is beta
prior_kappa = [0;1]; %Uniform pdf; First element is lower bound, second element is upper bound
prior_r_pi = [1.5;0.25]; %Normal pdf; First element is mean, second element is standard deviation
prior_r_x = [0.5;0.25]; %Normal pdf; First element is mean, second element is standard deviation
prior_sigma_x = [6;5]; %Inverse Gamma pdf; First element is shape parameter (alpha), second element is scale parameter (beta)
prior_sigma_pi = [6;5]; %Inverse Gamma pdf; First element is shape parameter (alpha), second element is scale parameter (beta)
prior_sigma_i = [6;5]; %Inverse Gamma pdf; First element is shape parameter (alpha), second element is scale parameter (beta)
prior_rho_x = [0;0.97]; %Uniform pdf; First element is lower bound, second element is upper bound
prior_rho_pi = [0;0.97]; %Uniform pdf; First element is lower bound, second element is upper bound

%% Rational Expectations state space matrices

Gamma_0 = [1,0,sigma^-1,-1,0,-1,-sigma^-1;
           -kappa,1,0,0,-1,0,-beta;
           -r_x,-r_pi,1,0,0,0,0;
           0,0,0,1,0,0,0;
           0,0,0,0,1,0,0;
           1,0,0,0,0,0,0;
           0,1,0,0,0,0,0];

Gamma_1 = [0,0,0,0,0,0,0;
           0,0,0,0,0,0,0;
           0,0,0,0,0,0,0;
           0,0,0,rho_x,0,0,0;
           0,0,0,0,rho_pi,0,0;
           0,0,0,0,0,1,0;
           0,0,0,0,0,0,1];
       
Psi = [0,0,0;
       0,0,0;
       0,0,sigma_i;
       sigma_x,0,0;
       0,sigma_pi,0;
       0,0,0;
       0,0,0];
   
Pi = [0,0;
      0,0;
      0,0;
      0,0;
      0,0;
      1,0;
      0,1];

C_re = zeros(variables_end+variables_exp,1);

%Find initial RE solution using gensys
[G1,~,impact,~,~,~,~,~]=gensys(Gamma_0,Gamma_1,C_re,Psi,Pi);

%Initialize mean and covariance matrix for KF
shat = k1.*zeros(variables_end+variables_exp,1); %Matrix of means of the prior dist. of the state
sig = k2.*diag(ones(variables_end+variables_exp,1)); %Covariance matrix of prior dist. of state

%Initial value for log-likelihood for KF
lhT = 0;

%Use KF to generate log-likelihood 
 for t=2:T %Start KF recursion
    [shatnew,signew,lh,~]=kf(data(t,:)',H_re,shat,sig,G1,impact,C_re,variables_end+variables_exp);
    lhT=(lhT+lh);
    shat = shatnew;
    sig = signew;
 end %End Kalman filter recursion
Llik = sum(lhT); %Value of log-likelihood

%% Evaluate prior at values of parameters
P_sigma = gampdf(sigma,prior_sigma(1),prior_sigma(2));
P_beta = betapdf(beta,prior_beta(1),prior_beta(2));
P_kappa = unifpdf(kappa,prior_kappa(1),prior_kappa(2));
P_r_pi = normpdf(r_pi,prior_r_pi(1),prior_r_pi(2));
P_r_x = normpdf(r_x,prior_r_x(1),prior_r_x(2));
P_sigma_x = invgamma_pdf(sigma_x,prior_sigma_x(1),prior_sigma_x(2));
P_sigma_pi = invgamma_pdf(sigma_pi,prior_sigma_pi(1),prior_sigma_pi(2));
P_sigma_i = invgamma_pdf(sigma_i,prior_sigma_i(1),prior_sigma_i(2));
P_rho_x = unifpdf(rho_x,prior_rho_x(1),prior_rho_x(2));
P_rho_pi = unifpdf(rho_pi,prior_rho_pi(1),prior_rho_pi(2));

Pl = log(P_sigma)+log(P_beta)+log(P_kappa)+log(P_r_pi)+log(P_r_x)+log(P_sigma_x)+log(P_sigma_pi)+log(P_sigma_i)+log(P_rho_x)+log(P_rho_pi);

%% Calculate value of log posterior
post = -(Llik + Pl);


end

