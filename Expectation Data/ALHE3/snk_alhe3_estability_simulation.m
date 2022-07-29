%%
%Simple New Keynesian Model with Adaptive Learning
%Heterogeneous Agent Expectations Equilibrium Simulation

clear
clc

%% Set seed state
seed = 123;
rng(seed);

%% Parameters
sigma_x = 1;        
sigma_pi = 1;        
sigma_i = 1;
sigma = 2;         
beta = 0.99;       
kappa = 0.5;       
i_x = 0.5;     
i_pi = 1.5;            
rho_x = 0.49;     
rho_pi = 0.49; 
alpha_A = 0.5;
alpha_B = 0;

% sigma_x = 0.8533;        
% sigma_pi = 1.0410;        
% sigma_i = 1.1814;
% sigma = 1.6289;         
% beta = 0.9925;       
% kappa = 0.5844;       
% i_x = 0.9070;     
% i_pi = 1.6429;            
% rho_x = 0.9192;     
% rho_pi = 0.7604; 
% alpha_A = 0.3397;
% alpha_B = 0.2534;

%Matrices
D1 = [1,0;-kappa,1];
D2 = [alpha_A-(sigma^-1)*i_x*alpha_A,-(sigma^-1)*i_pi*alpha_A+(sigma^-1)*alpha_A;0,beta*alpha_A];
D3 = [alpha_B-(sigma^-1)*i_x*alpha_B,-(sigma^-1)*i_pi*alpha_B+(sigma^-1)*alpha_B;0,beta*alpha_B];
D4 = [(1-alpha_A-alpha_B)-(sigma^-1)*i_x*(1-alpha_A-alpha_B),-(sigma^-1)*i_pi*(1-alpha_A-alpha_B)+(sigma^-1)*(1-alpha_A-alpha_B);0,beta*(1-alpha_A-alpha_B)];
D5 = [1;0];
D6 = [0;1];
D7 = [-(sigma^-1);0];

D1_inv = D1^-1;

F1 = D1_inv*D2;
F2 = D1_inv*D3;
F3 = D1_inv*D4;
F4 = D1_inv*D5;
F5 = D1_inv*D6;
F6 = D1_inv*D7;

n = size(D1,1);
I = eye(n);

%HEE Solution
G1 = [F1*rho_x-I,zeros(n),F2*rho_x,zeros(n);zeros(n),F1*rho_pi-I,zeros(n),F3*rho_pi;F1*rho_x,zeros(n),F2*rho_x-I,zeros(n);zeros(n),F1*rho_pi,zeros(n),F3*rho_pi-I];
G2 = [F4;F5;F4;F5];
Gamma_inv = G1^-1;
HEE = -Gamma_inv*G2;
 
A_x_bar = HEE(1:2);
A_pi_bar = HEE(3:4);
B_bar = HEE(5:6);
C_bar = HEE(7:8);

%% E-Stability
HEE_Estab = eig(G1);

%% RLS Algorithm

%Number of time periods
T = 1000;

%Create storage for exogenous variables
u = zeros(2,1,T+1);

%Create storage for endogenous variables
y = zeros(2,1,T+1);

%Create storage for beliefs
A_learning = zeros(2,2,T+1);
B_learning = zeros(2,1,T+1);
C_learning = zeros(2,1,T+1);

%Create storage for moment matrices
R_A_x = zeros(2,2,T+1); 
R_A_x(:,:,1) = eye(2);

R_B_x = zeros(1,1,T+1);
R_B_x(:,:,1) = 1;

R_C_x = zeros(1,1,T+1);
R_C_x(:,:,1) = 1;

R_A_pi = zeros(2,2,T+1); 
R_A_pi(:,:,1) = eye(2);

R_B_pi = zeros(1,1,T+1);
R_B_pi(:,:,1) = 1;

R_C_pi = zeros(1,1,T+1);
R_C_pi(:,:,1) = 1;

%Create storage for regressors
z_A = zeros(2,1,T+1);
z_B = zeros(1,1,T+1);
z_C = zeros(1,1,T+1);

%Create white noise values
epsilon_x = sigma_x*normrnd(0,1,T+1,1);
epsilon_pi = sigma_pi*normrnd(0,1,T+1,1);
epsilon_i = sigma_i*normrnd(0,1,T+1,1);

%Start adaptive learning
for t=2:T+1
    %Get values of exogenous variables one-period back
    u_x_tm1 = u(1,1,t-1);
    u_pi_tm1 = u(2,1,t-1);
    
    %Get values of endogenous variables one-period back (from ALM)
    x_tm1 = y(1,1,t-1); 
    pi_tm1 = y(2,1,t-1);

    %Get new regressors
    z_A_tm1 = [u_x_tm1;u_pi_tm1];
    z_B_tm1 = u_x_tm1;
    z_C_tm1 = u_pi_tm1;
    
    %Get previous values of beliefs
    A_x_tm1 = A_learning(1,:,t-1)'; 
    A_pi_tm1 = A_learning(2,:,t-1)'; 
    
    B_x_tm1 = B_learning(1,:,t-1)';
    B_pi_tm1 = B_learning(2,:,t-1)';
    
    C_x_tm1 = C_learning(1,:,t-1)';
    C_pi_tm1 = C_learning(2,:,t-1)';
    
    %Get previous values of moment matrices
    R_A_x_tm1 = R_A_x(:,:,t-1); 
    R_A_pi_tm1 = R_A_pi(:,:,t-1); 
    
    R_B_x_tm1 = R_B_x(:,:,t-1);
    R_B_pi_tm1 = R_B_pi(:,:,t-1);
    
    R_C_x_tm1 = R_C_x(:,:,t-1);
    R_C_pi_tm1 = R_C_pi(:,:,t-1);
    
    gn_A = t^-1;
    gn_B = t^-1;
    gn_C = t^-1;
    
%     gn_A = 0.1;
%     gn_B = 0.1;
%     gn_C = 0.1;
    
    %Update beliefs
    %Output gap
    [A_x_t,R_A_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_A,A_x_tm1,R_A_x_tm1,x_tm1,z_A_tm1);
    [B_x_t,R_B_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_B,B_x_tm1,R_B_x_tm1,x_tm1,z_B_tm1);
    [C_x_t,R_C_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_C,C_x_tm1,R_C_x_tm1,x_tm1,z_C_tm1);
        
    %Inflation
    [A_pi_t,R_A_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_A,A_pi_tm1,R_A_pi_tm1,pi_tm1,z_A_tm1);
    [B_pi_t,R_B_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_B,B_pi_tm1,R_B_pi_tm1,pi_tm1,z_B_tm1);
    [C_pi_t,R_C_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_C,C_pi_tm1,R_C_pi_tm1,pi_tm1,z_C_tm1);
    
    %Store new values of beliefs
    A_learning(1,:,t) = A_x_t'; %Output gap
    A_learning(2,:,t) = A_pi_t'; %Inflation
    
    B_learning(1,:,t) = B_x_t';
    B_learning(2,:,t) = B_pi_t';
    
    C_learning(1,:,t) = C_x_t';
    C_learning(2,:,t) = C_pi_t';
    
    A_x = [A_x_t(1);A_pi_t(1)];
    A_pi = [A_x_t(2);A_pi_t(2)];
    
    B = [B_x_t;B_pi_t];
    
    C = [C_x_t;C_pi_t];
    
        
    %Store regressors
    z_A(:,t) = z_A_tm1;
    z_B(:,t) = z_B_tm1;
    z_C(:,t) = z_C_tm1;
        
    %Store new values of the moment matrices
    R_A_x(:,:,t) = R_A_x_t; %Output
    R_A_pi(:,:,t) = R_A_pi_t; %Inflation
    
    R_B_x(:,:,t) = R_B_x_t;
    R_B_pi(:,:,t) = R_B_pi_t;
    
    R_C_x(:,:,t) = R_C_x_t;
    R_C_pi(:,:,t) = R_C_pi_t;
    
    %Get new values of exogenous variables
    u_x_t = rho_x*u_x_tm1 + epsilon_x(t);
    u_pi_t = rho_pi*u_pi_tm1 + epsilon_pi(t);
    u(1,1,t) = u_x_t;
    u(2,1,t) = u_pi_t;
    
    %Get new values of endogenous variables (from ALM)
    y(:,1,t) = (F1*A_x*rho_x + F2*B*rho_x + F4)*u_x_t + (F1*A_pi*rho_pi + F3*C*rho_pi + F5)*u_pi_t + F6*epsilon_i(t);
    
end
    
%% Creat plots
A_01_01 = squeeze(A_learning(1,1,2:end));
A_01_02 = squeeze(A_learning(1,2,2:end));
A_02_01 = squeeze(A_learning(2,1,2:end));
A_02_02 = squeeze(A_learning(2,2,2:end));

B_01 = squeeze(B_learning(1,1,2:end));
B_02 = squeeze(B_learning(2,1,2:end));

C_01 = squeeze(C_learning(1,1,2:end));
C_02 = squeeze(C_learning(2,1,2:end));

time = [1,1:T-1];


figure(1)
subplot(2,1,1)
plot(time,A_01_01)
title('$A_{1,1}$','Interpreter','latex')

subplot(2,1,2)
plot(time,A_02_01)
title('$A_{2,1}$','Interpreter','latex')

figure(2)
subplot(2,1,1)
plot(time,A_01_02)
title('$A_{1,2}$','Interpreter','latex')

subplot(2,1,2)
plot(time,A_02_02)
title('$A_{2,2}$','Interpreter','latex')

figure(3)
subplot(2,1,1)
plot(time,B_01)
title('$B_1$','Interpreter','latex')

subplot(2,1,2)
plot(time,B_02)
title('$B_2$','Interpreter','latex')

figure(4)
subplot(2,1,1)
plot(time,C_01)
title('$C_1$','Interpreter','latex')

subplot(2,1,2)
plot(time,C_02)
title('$C_2$','Interpreter','latex')

