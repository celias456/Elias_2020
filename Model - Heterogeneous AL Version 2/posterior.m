function [post] = posterior(parameters)
%Calculates the value of the posterior

%% Characteristics of model
variables_end = 5; %Number of endogenous variables
variables_exo = 3; %Number of exogenous variables
variables_exp = 2; %Number of expectational variables
num_parameters = 13; %Number of parameters to be estimated
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
gain_b = parameters(2);
alpha = parameters(3);
sigma = parameters(4);
beta = parameters(5);
kappa = parameters(6);
r_pi = parameters(7);
r_x = parameters(8);
sigma_x = parameters(9);
sigma_pi = parameters(10);
sigma_i = parameters(11);
rho_x = parameters(12);
rho_pi = parameters(13);

%% Priors

prior_gain_a = [0;0.15]; %Uniform pdf; First element is lower bound, second element is upper bound
prior_gain_b = [0;0.15]; %Uniform pdf; First element is lower bound, second element is upper bound
prior_alpha = [0;1]; %Uniform pdf; First element is lower bound, second element is upper bound
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

%Agent b
num_regressors_b = variables_state - 1 + 1;

%Initial values of beliefs from RE solution
%Output gap
d1_initial = 0; 
e1_initial = G1(1,4); 
%Inflation
d2_initial = 0; 
e2_initial = G1(2,4);

%Initial value for S matrices 
S = eye(num_regressors_b);

%Create storage for learning parameter coefficients and store initial
%values
S1 = zeros(num_regressors_b,num_regressors_b,T); %Output gap
S2 = zeros(num_regressors_b,num_regressors_b,T); %Inflation

S1(:,:,1) = S;
S2(:,:,1) = S;

phi_b_x = zeros(num_regressors_b,T); %Learning coefficients for output gap
phi_b_pi = zeros(num_regressors_b,T); %Learning coefficients for inflation

phi_b_x(:,1) = [d1_initial;e1_initial]; %Iinitial coefficients on output gap
phi_b_pi(:,1) = [d2_initial;e2_initial]; %Initial coefficients on Inflation

X_b = zeros(num_regressors_b,T);
X_b(:,1) = [1;0];

%Initialize AL state-space matrices using initial values of beliefs 
 
A0 = [1,0,sigma^-1,-(alpha*b1_initial*rho_x + sigma^-1*alpha*b2_initial*rho_x + 1), -(alpha*c1_initial*rho_pi + (1-alpha)*e1_initial*rho_pi + sigma^-1*alpha*c2_initial*rho_pi + sigma^-1*(1-alpha)*e2_initial*rho_pi);
      -kappa,1,0,-(beta*alpha*b2_initial*rho_x),-(beta*alpha*c2_initial*rho_pi + beta*(1-alpha)*e2_initial*rho_pi + 1);
      -r_x,-r_pi,1,0,0;
      0,0,0,1,0;
      0,0,0,0,1];

A1 = [alpha*a1_initial+(1-alpha)*d1_initial + sigma^-1*alpha*a2_initial + sigma^-1*(1-alpha)*d2_initial;
      beta*alpha*a2_initial + beta*(1-alpha)*d2_initial;
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
    
    %Agent b
    %Get regressors
    X_b(:,t) = [1;shat(5)];
    
    %Update beliefs
    %Output gap
    S1(:,:,t) = S1(:,:,t-1) + gain_b*(X_b(:,t-1)*X_b(:,t-1)' - S1(:,:,t-1));
    S1inv = S1(:,:,t)^-1;
    phi_b_x(:,t) = phi_b_x(:,t-1) + gain_b*S1inv*X_b(:,t)*(shatnew(1) - X_b(:,t)'*phi_b_x(:,t-1));
    
    d1 = phi_b_x(1,t);
    e1 = phi_b_x(2,t);
    
    %Inflation
    S2(:,:,t) = S2(:,:,t-1) + gain_b*(X_b(:,t-1)*X_b(:,t-1)' - S2(:,:,t-1));
    S2inv = S2(:,:,t)^-1;
    phi_b_pi(:,t) = phi_b_pi(:,t-1) + gain_b*S2inv*X_b(:,t)*(shatnew(2) - X_b(:,t)'*phi_b_pi(:,t-1));
    
    d2 = phi_b_pi(1,t);
    e2 = phi_b_pi(2,t);
     
    %Update AL state-space matrices using new values of beliefs 
    A0 = [1,0,sigma^-1,-(alpha*b1*rho_x + sigma^-1*alpha*b2*rho_x + 1), -(alpha*c1*rho_pi + (1-alpha)*e1*rho_pi + sigma^-1*alpha*c2*rho_pi + sigma^-1*(1-alpha)*e2*rho_pi);
       -kappa,1,0,-(beta*alpha*b2*rho_x),-(beta*alpha*c2*rho_pi + beta*(1-alpha)*e2*rho_pi + 1);
       -r_x,-r_pi,1,0,0;
       0,0,0,1,0;
       0,0,0,0,1];

    A1 = [alpha*a1+(1-alpha)*d1 + sigma^-1*alpha*a2 + sigma^-1*(1-alpha)*d2;
       beta*alpha*a2 + beta*(1-alpha)*d2;
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
P_gain_b = unifpdf(gain_b,prior_gain_b(1),prior_gain_b(2));
P_alpha = unifpdf(alpha,prior_alpha(1),prior_alpha(2));
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

Pl = log(P_gain_a)+log(P_gain_b)+log(P_alpha)+log(P_sigma)+log(P_beta)+log(P_kappa)+log(P_r_pi)+log(P_r_x)+log(P_sigma_x)+log(P_sigma_pi)+log(P_sigma_i)+log(P_rho_x)+log(P_rho_pi);

%% Calculate value of log posterior
post = -(Llik + Pl);


end

