%%
%Small Scale NK Model 
%Rational Expectations
%Likelihood calculation and impulse responses

clear
clc

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

%% Initial values of parameters
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

parameters = [sigma,beta,kappa,r_pi,r_x,sigma_x,sigma_pi,sigma_i,rho_x,rho_pi];

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
[G1,~,impact,~,~,~,~,eu]=gensys(Gamma_0,Gamma_1,C_re,Psi,Pi);

%Initialize mean and covariance matrix for KF
shat = k1.*zeros(variables_end+variables_exp,1); %Matrix of means of the prior dist. of the state
sig = k2.*diag(ones(variables_end+variables_exp,1)); %Covariance matrix of prior dist. of state

%Initial value for log-likelihood for KF
lhT = 0;

%Use KF to generate log-likelihood 
 for t=2:T %Start KF recursion
    [shatnew,signew,lh,yhat]=kf(data(t,:)',H_re,shat,sig,G1,impact,C_re,variables_end+variables_exp);
    lhT=(lhT+lh);
    shat = shatnew;
    sig = signew;
 end %End Kalman filter recursion
Llik = sum(lhT); %Value of log-likelihood
L = exp(Llik); %Value of likelihood

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

post = Llik + Pl;

%% Minimize negative of log posterior
% A = [];
% b = [];
% Aeq = [];
% beq = [];
% lb = [0,0,0,-100,-100,0,0,0,0,0];
% ub = [100,0.9999,100,100,100,100,100,100,0.97,0.97];
% options = optimset('Algorithm','active-set');
% %Assign function handle
% f = @(input)posterior(input);
% %Perform minimization
% [theta_hat,value_theta_hat] = fmincon(f,parameters,A,b,Aeq,beq,lb,ub,options);


%% Impulse response functions

% col = impact;
% periods=40;
% resp = zeros(7,3,40);
% for j=1:periods
% resp(:,:,j)=col; % Stores the i-th response of the variables to the shocks.
% col=G1*col; % Multiplies by G1 to give the next step response to the
% % shocks.
% end
% % "squeeze" eliminates the singleton dimensions
% % of resp(:,:,:). It returns a matrix with the first ten
% % responses of the 1st variable to the 1st shock
% resp1y_RE(:,1)=squeeze(resp(1,1,:)); %Response of output gap to preference shock
% resp1pi_RE(:,1)=squeeze(resp(2,1,:)); %Response of inflation to preference shock
% resp1i_RE(:,1)=squeeze(resp(3,1,:)); %Response of interest rate to preference shock 
% resp2y_RE(:,1)=squeeze(resp(1,2,:)); %Response of output gap to cost push shock
% resp2pi_RE(:,1)=squeeze(resp(2,2,:)); %Response of inflation to cost push shock
% resp2i_RE(:,1)=squeeze(resp(3,2,:)); %Response of interest rate to cost push shock 
% resp3y_RE(:,1)=squeeze(resp(1,3,:)); %Response of output gap to monetary policy shock
% resp3pi_RE(:,1)=squeeze(resp(2,3,:)); %Response of inflation to monetary policy shock
% resp3i_RE(:,1)=squeeze(resp(3,3,:)); %Response of interest rate to monetary policy shock 
% 
% figure(1)
% subplot(3,3,1)
% plot(1:periods,resp1y_RE(:,1))
% title('Output Gap to Preference Shock'); grid on
% subplot(3,3,2)
% plot(1:periods,resp1pi_RE(:,1));
% title('Inflation to Preference Shock'); grid on
% subplot(3,3,3)
% plot(1:periods,resp1i_RE(:,1));
% title('Interest Rate to Preference Shock'); grid on
% subplot(3,3,4)
% plot(1:periods,resp2y_RE(:,1))
% title('Output Gap to Cost Push Shock'); grid on
% subplot(3,3,5)
% plot(1:periods,resp2pi_RE(:,1))
% title('Inflation to Cost Push Shock'); grid on
% subplot(3,3,6)
% plot(1:periods,resp2i_RE(:,1));
% title('Interest Rate to Cost Push Shock'); grid on
% subplot(3,3,7)
% plot(1:periods,resp3y_RE(:,1));
% title('Output Gap to Monetary Policy Shock'); grid on
% subplot(3,3,8)
% plot(1:periods,resp3pi_RE(:,1))
% title('Inflation to Monetary Policy Shock'); grid on
% subplot(3,3,9)
% plot(1:periods,resp3i_RE(:,1))
% title('Interest Rate to Monetary Policy Shock'); grid on
