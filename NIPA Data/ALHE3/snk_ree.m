%%
%Simple New Keynesian Model with Adaptive Learning
%Rational Expectations Equilibrium
%Main File

clear
clc

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

%Matrices
C1 = [1,0;-kappa,1];
C2 = [1-(sigma^-1)*i_x,(sigma^-1)-(sigma^-1)*i_pi;0,beta];
C3 = [1;0];
C4 = [0;1];
C5 = [-sigma^-1;0];

C1_inv = C1^-1;

A = C1_inv*C2;
B = C1_inv*C3;
C = C1_inv*C4;
D = C1_inv*C5;

n = size(C1,1);
I = eye(n);

%RE Solution
P1_bar = ((I-A*rho_x)^-1)*B;
P2_bar = ((I-A*rho_pi)^-1)*C;