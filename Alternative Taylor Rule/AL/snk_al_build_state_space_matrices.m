function [J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2] = snk_al_build_state_space_matrices(number_endogenous_variables,number_jumper_variables_agent_type,number_exogenous_variables,number_observed_variables,theta,A,B,C)
%Builds the adaptive learning state space matrices of the simple New
%Keynesian model with adaptive learning (AL)
%number_endogenous_variables: Number of endogenous variables
%number_jumper_variables_agent_type: Number of agent-type jumper variables
%number_exogenous_variables: Number of exogenous variables
%number_observed_variables: Number of exogenous variables (stochastic shocks)
%theta: Column vector of parameters
%A: Matrix of agent beliefs
%B: Matrix of agent beliefs (agent-type B)
%C: Matrix of agent beliefs (agent-type C)

%J_1 and J_2 are the matrices of the state (plant) equation:
%   s_t = J_1*s_{t-1} + J_2*varepsilon_t
%
%Psi_L_0, Psi_L_1, and Psi_L_2 are the matrices of the measurement (observation) equation:
%   y_t = Psi_L_0 + Psi_L_1*t + Psi_L_2*s_t + u_t

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get system information
%Number of variables in state transition equation
n = number_endogenous_variables + number_jumper_variables_agent_type;

%Fixed parameters
alpha_A = 1;
alpha_B = 0;

%Estimated parameters
sigma_x = theta(1);        
sigma_pi = theta(2);        
sigma_i = theta(3);
sigma = theta(4);         
beta = theta(5);       
kappa = theta(6);       
i_x = theta(7);     
i_pi = theta(8);            
rho_x = theta(9);     
rho_pi = theta(10);

%Adaptive learning beliefs
%Agent-type A
A_01_01 = A(1,1);
A_01_02 = A(1,2);
A_02_01 = A(2,1);
A_02_02 = A(2,2);

%Agent-type B
B_01 = B(1,1);
B_02 = B(2,1);

%Agent-type C
C_01 = C(1,1);
C_02 = C(2,1);

%Composite parameters

%Stochastic shocks
epsilon_standard_deviations = [sigma_x;sigma_pi;sigma_i];

%Initialization for transformed J2 matrix
J_2 = zeros(n,number_exogenous_variables);

%% Equation indexes

%State transition equation
eq_s_1 = 1;  %IS Curve
eq_s_2 = 2;  %New Keynesian Phillips curve
eq_s_3 = 3;  %Monetary policy rule
eq_s_4 = 4;  %Law of motion for aggregate demand shock process
eq_s_5 = 5;  %Law of motion for aggregate supply shock process
eq_s_6 = 6;  %Agent-type A expected output gap
eq_s_7 = 7;  %Agent-type B expected output gap
eq_s_8 = 8;  %Agent-type C expected output gap
eq_s_9 = 9;  %Agent-type A expected inflation
eq_s_10 = 10;  %Agent-type B expected inflation
eq_s_11 = 11;  %Agent-type C expected inflation

%Measurement equations
eq_m_1 = 1;  %Output gap
eq_m_2 = 2;  %Inflation
eq_m_3 = 3;  %Interest rate


%% Variable indexes in state transition equation

x = 1;          %Output gap
pi = 2;         %Inflation
i = 3;          %Nominal interest rate
u_x = 4;        %Aggregate demand shock process
u_pi = 5;       %Aggregate supply shock process
E_A_x = 6;      %Agent-type A expected output gap
E_B_x = 7;      %Agent-type B expected output gap
E_C_x = 8;      %Agent-type C expected output gap
E_A_pi = 9;     %Agent-type A expected inflation
E_B_pi = 10;    %Agent-type B expected inflation
E_C_pi = 11;    %Agent-type C expected inflation

%% Stochastic shock indexes (epsilon)

varepsilon_x = 1; %Aggregate demand shock
varepsilon_pi = 2; %Aggregate supply shock
varepsilon_i = 3; %Monetary policy shock

%% Initialize Matrices

%State transition equation
H_1 = zeros(n,n);
H_2 = zeros(n,n);
H_3 = zeros(n,number_exogenous_variables);

%Measurement equation
Psi_L_0 = zeros(number_observed_variables,1);
Psi_L_1 = zeros(number_observed_variables,number_observed_variables);
Psi_L_2 = zeros(number_observed_variables,n);

%% Put system into matrix form

%Equation 1
H_1(eq_s_1,x) = 1;
H_1(eq_s_1,i) = sigma^-1;
H_1(eq_s_1,u_x) = -1;
H_1(eq_s_1,E_A_x) = -alpha_A;
H_1(eq_s_1,E_B_x) = -alpha_B;
H_1(eq_s_1,E_C_x) = -(1-alpha_A-alpha_B);
H_1(eq_s_1,E_A_pi) = -(sigma^-1)*alpha_A;
H_1(eq_s_1,E_B_pi) = -(sigma^-1)*alpha_B;
H_1(eq_s_1,E_C_pi) = -(sigma^-1)*(1-alpha_A-alpha_B);

%Equation 2
H_1(eq_s_2,x) = -kappa;
H_1(eq_s_2,pi) = 1;
H_1(eq_s_2,u_pi) = -1;
H_1(eq_s_2,E_A_pi) = -beta*alpha_A;
H_1(eq_s_2,E_B_pi) = -beta*alpha_B;
H_1(eq_s_2,E_C_pi) = -beta*(1-alpha_A-alpha_B);

%Equation 3
H_1(eq_s_3,x) = -i_x;
H_1(eq_s_3,pi) = -i_pi;
H_1(eq_s_3,i) = 1;

H_3(eq_s_3,varepsilon_i) = sigma_i;

%Equation 4
H_1(eq_s_4,u_x) = 1;

H_2(eq_s_4,u_x) = rho_x;

H_3(eq_s_4,varepsilon_x) = sigma_x;

%Equation 5
H_1(eq_s_5,u_pi) = 1;

H_2(eq_s_5,u_pi) = rho_pi;

H_3(eq_s_5,varepsilon_pi) = sigma_pi;

%Equation 6
H_1(eq_s_6,u_x) = -A_01_01*rho_x;
H_1(eq_s_6,u_pi) = -A_01_02*rho_pi;
H_1(eq_s_6,E_A_x) = 1;

%Equation 7
H_1(eq_s_7,u_x) = -B_01*rho_x;
H_1(eq_s_7,E_B_x) = 1;

%Equation 8
H_1(eq_s_8,u_pi) = -C_01*rho_pi;
H_1(eq_s_8,E_C_x) = 1;

%Equation 9
H_1(eq_s_9,u_x) = -A_02_01*rho_x;
H_1(eq_s_9,u_pi) = -A_02_02*rho_pi;
H_1(eq_s_9,E_A_pi) = 1;

%Equation 10
H_1(eq_s_10,u_x) = -B_02*rho_x;
H_1(eq_s_10,E_B_pi) = 1;

%Equation 11
H_1(eq_s_11,u_pi) = -C_02*rho_pi;
H_1(eq_s_11,E_C_pi) = 1;

%% Generate state transition equation by solving for ALM

%Test for singularity
test_H_1 = det(H_1);

if test_H_1 == 0 %H_1 is singular; Take the p-inverse
   H1_inv = pinv(H_1);
else
   H1_inv = H_1^(-1);
end

J_1 = H1_inv*H_2;
J_c = zeros(n,1); %Constants; all zeros in this model
J_2_pre_transformation = H1_inv*H_3;

%% Transform J2 matrix

for index_1 = 1:number_exogenous_variables
    J_2(:,index_1) = J_2_pre_transformation(:,index_1)/epsilon_standard_deviations(index_1); %Transformed J_2 matrix
end

%% Measurement equation

%Equation 1
Psi_L_2(eq_m_1,x) = 1;

%Equation 2
Psi_L_2(eq_m_2,pi) = 1;

%Equation 3
Psi_L_2(eq_m_3,i) = 1;


end

