function [post] = posterior(parameters)
%Calculates the value of the posterior

%% Characteristics of model
variables_end = 5; %Number of endogenous variables
variables_exo = 3; %Number of exogenous variables
variables_exp = 2; %Number of expectational variables
num_parameters = 11; %Number of parameters to be estimated
variables_state = 2; %Number of state variables

%% Data 

data = csvread('Data.csv'); 
    
H_al = [1,0,0,0,0;
        0,1,0,0,0;
        0,0,1,0,0];
    
T = size(data,1);
    
%% Initial values for KF
k1 = 1; %Mean of prior distribution of state
k2 = 3; %Variance of prior distribution of state
    
%% Values of parameters

gain_a = parameters(1);
sigma = parameters(2);
beta = parameters(3);
kappa = parameters(4);
r_pi = parameters(5);
r_x = parameters(6);
sigma_x = parameters(7);
sigma_pi = parameters(8);
sigma_i = parameters(9);
rho_x = parameters(10);
rho_pi = parameters(11);

%% Priors
prior_gain_a = [0;0.15]; %Uniform pdf; First element is lower bound, second element is upper bound
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
[G1,~,~,~,~,~,~,~]=gensys(Gamma_0,Gamma_1,C_re,Psi,Pi);

%% Adaptive learning characteristics

%Agent a
num_regressors_a = variables_state + 1;

%Initial values of beliefs from RE solution

%Output gap
a1_initial = 0; 
b1_initial = G1(1,4); 
c1_initial = G1(1,5);
%Inflation
a2_initial = 0; 
b2_initial = G1(2,4); 
c2_initial = G1(2,5); 

%Initial value for R matrices
R = eye(num_regressors_a); 

%Create storage for learning parameter coefficients and store initial
%values
R1 = zeros(num_regressors_a,num_regressors_a,T); %Output gap
R2 = zeros(num_regressors_a,num_regressors_a,T); %Inflation

R1(:,:,1) = R;
R2(:,:,1) = R;

phi_a_x = zeros(num_regressors_a,T); %Learning coefficients for output gap
phi_a_pi = zeros(num_regressors_a,T); %Learning coefficients for inflation

phi_a_x(:,1) = [a1_initial;b1_initial;c1_initial]; %Iinitial coefficients on output gap
phi_a_pi(:,1) = [a2_initial;b2_initial;c2_initial]; %Initial coefficients on Inflation

X_a = zeros(num_regressors_a,T);
X_a(:,1) = [1;0;0];

%Initialize AL state-space matrices using initial values of beliefs 
 
A0 = [1,0,sigma^-1,-(b1_initial*rho_x + sigma^-1*b2_initial*rho_x + 1), -(c1_initial*rho_pi + sigma^-1*c2_initial*rho_pi);
      -kappa,1,0,-(beta*b2_initial*rho_x),-(beta*c2_initial*rho_pi + 1);
      -r_x,-r_pi,1,0,0;
      0,0,0,1,0;
      0,0,0,0,1];

A1 = [a1_initial + sigma^-1*a2_initial;
      beta*a2_initial;
      0;
      0;
      0];

A2 = [0,0,0,0,0;
      0,0,0,0,0;
      0,0,0,0,0;
      0,0,0,rho_x,0;
      0,0,0,0,rho_pi];

A3 = [0,0,0;
      0,0,0;
      0,0,sigma_i;
      sigma_x,0,0;
      0,sigma_pi,0];

A0_inv = A0^-1;
C_al = A0_inv*A1;
G = A0_inv*A2;
M = A0_inv*A3;

%Initialize mean and covariance matrix for KF
shat = k1.*zeros(variables_end,1); %Matrix of means of the prior dist. of the state
sig = k2.*diag(ones(variables_end,1)); %Covariance matrix of prior dist. of state

%Initial value for log-likelihood for KF
lhT = 0; 
lh = 0;

%Use KF to generate likelihood
 for t=2:T %Start KF recursion
    [shatnew,signew,lh,~]=kf(data(t,:)',H_al,shat,sig,G,M,C_al,variables_end);
    lhT=(lhT+lh);
 
    %Agent a
    %Get regressors
    X_a(:,t) = [1;shat(4);shat(5)];
      
    %Update beliefs
    %Output gap
    R1(:,:,t) = R1(:,:,t-1) + gain_a*(X_a(:,t-1)*X_a(:,t-1)' - R1(:,:,t-1));
    R1inv = R1(:,:,t)^-1;
    phi_a_x(:,t) = phi_a_x(:,t-1) + gain_a*R1inv*X_a(:,t)*(shatnew(1) - X_a(:,t)'*phi_a_x(:,t-1));
    
    a1 = phi_a_x(1,t);
    b1 = phi_a_x(2,t);
    c1 = phi_a_x(3,t);
    
    %Inflation
    R2(:,:,t) = R2(:,:,t-1) + gain_a*(X_a(:,t-1)*X_a(:,t-1)' - R2(:,:,t-1));
    R2inv = R2(:,:,t)^-1;
    phi_a_pi(:,t) = phi_a_pi(:,t-1) + gain_a*R2inv*X_a(:,t)*(shatnew(2) - X_a(:,t)'*phi_a_pi(:,t-1));
    
    a2 = phi_a_pi(1,t);
    b2 = phi_a_pi(2,t);
    c2 = phi_a_pi(3,t);
     
    %Update AL state-space matrices using new values of beliefs 
    A0 = [1,0,sigma^-1,-(b1*rho_x + sigma^-1*b2*rho_x + 1), -(c1*rho_pi + sigma^-1*c2*rho_pi);
          -kappa,1,0,-(beta*b2*rho_x),-(beta*c2*rho_pi + 1);
          -r_x,-r_pi,1,0,0;
          0,0,0,1,0;
          0,0,0,0,1];

    A1 = [a1 + sigma^-1*a2;
          beta*a2;
          0;
          0;
          0];

    A2 = [0,0,0,0,0;
          0,0,0,0,0;
          0,0,0,0,0;
          0,0,0,rho_x,0;
          0,0,0,0,rho_pi];

    A3 = [0,0,0;
          0,0,0;
          0,0,sigma_i;
          sigma_x,0,0;
          0,sigma_pi,0];

    A0_inv = A0^-1;
    C_al = A0_inv*A1;
    G = A0_inv*A2;
    M = A0_inv*A3;
    
    %Get new estimates of shat and sig
    shat = shatnew;
    sig = signew;
    
 end %End Kalman filter recursion
Llik = sum(lhT);

%% Evaluate prior at values of parameters
P_gain_a = unifpdf(gain_a,prior_gain_a(1),prior_gain_a(2));
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

Pl = log(P_gain_a)+log(P_sigma)+log(P_beta)+log(P_kappa)+log(P_r_pi)+log(P_r_x)+log(P_sigma_x)+log(P_sigma_pi)+log(P_sigma_i)+log(P_rho_x)+log(P_rho_pi);

%% Calculate value of log posterior
post = -(Llik + Pl);


end

