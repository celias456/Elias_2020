%%
%Small scale NK model 
%Homogeneous agent adaptive learning
%Bayesian estimation

clear
clc
warning('off','all')

%% Set seed state
seed = 123;
rng(seed); 

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

%% MH characteristics

D = 300000; %Number of MH draws
burnrate = 0.2; %Proportion of draws to discard in MH chain
scaling = 0.003; %Increase this to have a lower acceptance rate in the MH algorithm
tau = 0.01; %1-tau quantile of Chi-squared distribution used in marginal likelihood calculation

%Standard deviation of noise term in MH algorithm
Sigma_mh = scaling*1; 

%Create holders of MH draws
theta = zeros(D,num_parameters);
theta_star = zeros(D,num_parameters);
theta_P = zeros(D,1); %hold the log prior associated with each draw
theta_L = zeros(D,1); %Hold the log likelihood associated with each draw
   
%Diagnostics
acceptance = zeros(D,1);

%% Initial values of parameters
gain_a = 0.001;
sigma = 0.8861; 
beta = 1;
kappa = 1;
r_pi = 1.8391;
r_x = 0.1764;
sigma_x = 0.5333;
sigma_pi = 1.1854;
sigma_i = 3.1612;
rho_x = 0.9541;
rho_pi = 0.9372;

%Set initial value of MH draws to initial value of parameters
theta(1,:) = [gain_a,sigma,beta,kappa,r_pi,r_x,sigma_x,sigma_pi,sigma_i,rho_x,rho_pi];

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


%% Solve model under RE with initial parameter values

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

%Determine if a unique RE solution exists with initial value of parameters
%Get G1 and impact matrices from RE solution
[G1,~,impact,~,~,~,~,eu]=gensys(Gamma_0,Gamma_1,C_re,Psi,Pi);

if sum(eu) ~= 2
   disp('Initial values of parameter do not produce a unique and stable RE solution!')
end

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

%% Start MH algorithm

for j = 2:D
    
%Get previous values of parameters
gain_a = theta(j-1,1);
sigma = theta(j-1,2); 
beta = theta(j-1,3);
kappa = theta(j-1,4);
r_pi = theta(j-1,5);
r_x = theta(j-1,6);
sigma_x = theta(j-1,7);
sigma_pi = theta(j-1,8);
sigma_i = theta(j-1,9);
rho_x = theta(j-1,10);
rho_pi = theta(j-1,11);

%Evaluate prior at previous values of parameters
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
P = P_gain_a*P_sigma*P_beta*P_kappa*P_r_pi*P_r_x*P_sigma_x*P_sigma_pi*P_sigma_i*P_rho_x*P_rho_pi;

%Get value of likelihood using KF with previous values of parameters

%Initialize values for first iteration of KF
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

%Initial value for likelihood for KF
lhT = 0;
lh = 0;
    
%Use KF to generate likelihood with previous values of parameters
for t=2:T %Start KF recursion
    [shatnew,signew,lh,yhat]=kf(data(t,:)',H_al,shat,sig,G,M,C_al,variables_end);
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
L = exp(Llik); %Value of likelihood with previous values of parameters

%Done with getting value of likelihood for previous values of
%parameters

%Generate candidate values of parameters
theta_star(j,:) = theta(j-1,:) + normrnd(0,Sigma_mh,1,num_parameters);

gain_a = theta_star(j,1);
sigma = theta_star(j,2); 
beta = theta_star(j,3);
kappa = theta_star(j,4);
r_pi = theta_star(j,5);
r_x = theta_star(j,6);
sigma_x = theta_star(j,7);
sigma_pi = theta_star(j,8);
sigma_i = theta_star(j,9);
rho_x = theta_star(j,10);
rho_pi = theta_star(j,11);

%Test for determinancy

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

%Determine if a unique RE solution exists with candidate values of parameters
[~,~,~,~,~,~,~,eu]=gensys(Gamma_0,Gamma_1,C_re,Psi,Pi);

if sum(eu) ~= 2 %Indeterminancy of RE solution with candidate values of parameters
   L_star = 0;
   P_star = 0;
else %RE solution is unique and stable; Get likelihood with candidate values of parameters using KF

%Evaluate priors at the candidate values of parameters
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

Pl_star = log(P_gain_a)+log(P_sigma)+log(P_beta)+log(P_kappa)+log(P_r_pi)+log(P_r_x)+log(P_sigma_x)+log(P_sigma_pi)+log(P_sigma_i)+log(P_rho_x)+log(P_rho_pi);
P_star = P_gain_a*P_sigma*P_beta*P_kappa*P_r_pi*P_r_x*P_sigma_x*P_sigma_pi*P_sigma_i*P_rho_x*P_rho_pi;  

%Get value of likelihood using KF with candidate values of parameters

%Initialize values for first iteration of KF
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

%Use KF to generate likelihood with candidate values of parameters
for t=2:T %Start KF recursion
    [shatnew,signew,lh,yhat]=kf(data(t,:)',H_al,shat,sig,G,M,C_al,variables_end);
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
Llik_star = sum(lhT);
L_star = exp(Llik_star); %Value of likelihood with candidate values of parameters
end

%Determine if candidate values of parameters are accepted
U = unifrnd(0,1);
testnum = L_star*P_star;
testden = L*P;
test = testnum/testden;
if isnan(test) == 1
    test_candidate = 0;
else
test_candidate = min(test,1);
end
if U <= test_candidate
   theta(j,:) = theta_star(j,:);
   theta_P(j) = Pl_star;
   theta_L(j) = Llik_star;
   acceptance(j) = 1;
else
   theta(j,:) = theta(j-1,:);
   theta_P(j) = Pl;
   theta_L(j) = Llik;
end

end

%Remove the burn-in draws
theta_burn = theta(burnrate*D+1:end,:);
theta_P_burn = theta_P(burnrate*D+1:end,:);
theta_L_burn = theta_L(burnrate*D+1:end,:);

%%
%Point estimates
estimates_point(1,:) = mean(theta_burn);
estimates_point(2,:) = median(theta_burn);
estimates_point(3,:) = mode(theta_burn);

%Interval estimates
%95 percent posterior probability interval
estimates_interval(1,:) = prctile(theta_burn,97.5);
estimates_interval(2,:) = prctile(theta_burn,2.5);

%Marginal likelihood estimate
ml = mlike(theta_burn,theta_P_burn,theta_L_burn,tau);

%Diagnostics
arate = sum(acceptance)/D;

subplot(4,4,1), hist(theta_burn(:,1));%Histogram of the posterior dist. of gain_a
title('Posterior of $g_a$','interpreter','latex')
subplot(4,4,2), hist(theta_burn(:,2));%Histogram of the posterior dist. of sigma
title('Posterior of $\sigma$','interpreter','latex')
subplot(4,4,3), hist(theta_burn(:,3));%Histogram of the posterior dist. of beta
title('Posterior of $\beta$','interpreter','latex')
subplot(4,4,4), hist(theta_burn(:,4));%Histogram of the posterior dist. of kappa
title('Posterior of $\kappa$','interpreter','latex')
subplot(4,4,5), hist(theta_burn(:,5));%Histogram of the posterior dist. of r_pi
title('Posterior of $r_{\pi}$','interpreter','latex')
subplot(4,4,6), hist(theta_burn(:,6));%Histogram of the posterior dist. of r_x
title('Posterior of $r_x$','interpreter','latex')
subplot(4,4,7), hist(theta_burn(:,7));%Histogram of the posterior dist. of sigma_x
title('Posterior of $\sigma_x$','interpreter','latex')
subplot(4,4,8), hist(theta_burn(:,8));%Histogram of the posterior dist. of sigma_pi
title('Posterior of $\sigma_{\pi}$','interpreter','latex')
subplot(4,4,9), hist(theta_burn(:,9));%Histogram of the posterior dist. of sigma_i
title('Posterior of $\sigma_i$','interpreter','latex')
subplot(4,4,10), hist(theta_burn(:,10));%Histogram of the posterior dist. of rho_x
title('Posterior of $\rho_x$','interpreter','latex')
subplot(4,4,11), hist(theta_burn(:,11));%Histogram of the posterior dist. of rho_pi
title('Posterior of $\rho_{\pi}$','interpreter','latex')

 


 

 