function [Phi_1,Phi_c,Phi_epsilon,Psi_0,Psi_1,Psi_2,t,Sigma_epsilon,solution] = snk_re_build_state_space_matrices(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,theta)
%Builds the state space matrices of the small scale New Keynesian model
%with rational expectations

%Input:
%number_endogenous_variables: Number of endogenous variables
%number_jumper_variables: Number of jumper variables
%number_exogenous_variables: Number of exogenous variables
%number_observed_variables: Number of exogenous variables (stochastic shocks)
%theta: Column vector of parameters

%Phi_1, Phi_c, and Phi_epsilon are the matrices of the state (plant) equation:
%   s_t = Phi_1*s_{t-1} + Phi_c + Phi_epsilon*epsilon_t
%
%Psi_0, Psi_1, and Psi_2 are the matrices of the measurement (observation) equation:
%   y_t = Psi_0 + Psi_1*t + Psi_2*s_t + u_t
%
%t: Vector representing trend values in the measurement equation
%Sigma_epsilon: Covariance matrix for the stochastic shocks
%solution: an indicator of the RE solution; 1 is a unique and stable solution, 0 means the solution is not unique and stable

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get system information
%Number of variables in state transition equation
n = number_endogenous_variables+number_jumper_variables;

%Parameters
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

%Composite parameters

%Stochastic shocks
epsilon = [sigma_x;sigma_pi;sigma_i];
Sigma_epsilon = zeros(number_exogenous_variables,number_exogenous_variables); %Initialization for stochastic shock covariance matrix

%Initialization for transformed Phi_epsilon matrix
Phi_epsilon = zeros(n,number_exogenous_variables);

%Vector representing trend values in the measurement equation
t = ones(number_observed_variables,1);

%% Equation indexes

%State transition equation
eq_s_1 = 1;  %IS Curve
eq_s_2 = 2;  %New Keynesian Phillips curve
eq_s_3 = 3;  %Monetary policy rule
eq_s_4 = 4;  %Law of motion for aggregate demand shock process
eq_s_5 = 5;  %Law of motion for aggregate supply shock process
eq_s_6 = 6;  %Expectational equation for output
eq_s_7 = 7;  %Expectational equation for inflation

%Measurement equations
eq_m_1 = 1;  %Output gap
eq_m_2 = 2;  %Inflation
eq_m_3 = 3;  %Nominal interest rate


%% Variable indexes in state transition equation

x = 1;          %Output gap
pi = 2;         %Inflation
i = 3;          %Nominal interest rate
u_x = 4;        %Aggregate demand shock process
u_pi = 5;       %Aggregate supply shock process
E_x = 6;        %Output jumper variable
E_pi = 7;       %Inflation jumper variable


%% Stochastic shock indexes (epsilon)

varepsilon_x = 1; %Aggregate demand shock
varepsilon_pi = 2; %Aggregate supply shock
varepsilon_i = 3; %Monetary policy shock

%% Expectation error indexes (eta)

eta_x = 1; %Output
eta_pi = 2; %Inflation

%% Initialize Matrices

%State transition equation
Gamma_0 = zeros(n,n);
Gamma_1 = zeros(n,n);
Gamma_c = zeros(n,1);
Psi = zeros(n,number_exogenous_variables);
Pi = zeros(n,number_jumper_variables);

%Measurement equation
Psi_0 = zeros(number_observed_variables,1);
Psi_1 = zeros(number_observed_variables,number_observed_variables);
Psi_2 = zeros(number_observed_variables,n);

%% Canonical System

%Equation 1
Gamma_0(eq_s_1,x) = 1;
Gamma_0(eq_s_1,i) = sigma^-1;
Gamma_0(eq_s_1,u_x) = -1;
Gamma_0(eq_s_1,E_x) = -1;
Gamma_0(eq_s_1,E_pi) = -sigma^-1;

%Equation 2
Gamma_0(eq_s_2,x) = -kappa;
Gamma_0(eq_s_2,pi) = 1;
Gamma_0(eq_s_2,u_pi) = -1;
Gamma_0(eq_s_2,E_pi) = -beta;

%Equation 3
Gamma_0(eq_s_3,i) = 1;
Gamma_0(eq_s_3,E_x) = -i_x;
Gamma_0(eq_s_3,E_pi) = -i_pi;

Psi(eq_s_3,varepsilon_i) = sigma_i;

%Equation 4
Gamma_0(eq_s_4,u_x) = 1;

Gamma_1(eq_s_4,u_x) = rho_x;

Psi(eq_s_4,varepsilon_x) = sigma_x;

%Equation 5
Gamma_0(eq_s_5,u_pi) = 1;

Gamma_1(eq_s_5,u_pi) = rho_pi;

Psi(eq_s_5,varepsilon_pi) = sigma_pi;

%Equation 6
Gamma_0(eq_s_6,x) = 1;

Gamma_1(eq_s_6,E_x) = 1;

Pi(eq_s_6,eta_x) = 1;

%Equation 7
Gamma_0(eq_s_7,pi) = 1;

Gamma_1(eq_s_7,E_pi) = 1;

Pi(eq_s_7,eta_pi) = 1;

%% Generate state transition equation by solving for RE solution
[Phi_1,Phi_c,Phi_epsilon_pre_transformation,~,~,~,~,eu] = gensys(Gamma_0,Gamma_1,Gamma_c,Psi,Pi);
solution_sum = sum(eu);

solution = 0;

if solution_sum == 2
    solution = 1;
end

%% Covariance matrix for the stochastic shocks and transform Phi_epsilon matrix

for index_1 = 1:number_exogenous_variables
    Sigma_epsilon(index_1,index_1) = epsilon(index_1)^2; %Covariance matrix for stochastic shocks
    Phi_epsilon(:,index_1) = Phi_epsilon_pre_transformation(:,index_1)/epsilon(index_1); %Transformed Phi_epsilon matrix
end

%% Measurement equation

%Equation 1
Psi_2(eq_m_1,x) = 1;

%Equation 2
Psi_2(eq_m_2,pi) = 1;

%Equation 3
Psi_2(eq_m_3,i) = 1;


end

