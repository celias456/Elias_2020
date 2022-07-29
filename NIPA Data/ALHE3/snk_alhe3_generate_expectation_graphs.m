clear
clc

%% Characteristics of model

number_endogenous_variables = 5; %Number of endogenous variables
number_jumper_variables = 2; %Number of jumper variables
number_exogenous_variables = 3; %Number of exogenous variables
number_observed_variables = 3; %Number of observable variables
number_state_variables = 2; %Number of state variables
number_plms = 2; %Number of perceived laws of motion

%% Values of Parameters

sigma_x = 0.505;        
sigma_pi = 1.010;        
sigma_i = 1.156;
sigma = 1.256;         
beta = 0.990;       
kappa = 0.982;       
i_x = 0.778;     
i_pi = 1.805;            
rho_x = 0.949;     
rho_pi = 0.893; 
gn_A = 0.015;
gn_B = 0.065;
gn_C = 0.054;
alpha_A = 0.640;
alpha_B = 0.037;

theta = [sigma_x;sigma_pi;sigma_i;sigma;beta;kappa;i_x;i_pi;rho_x;rho_pi;gn_A;gn_B;gn_C;alpha_A;alpha_B];

%Number of variables in rational expectations state vector
n = number_endogenous_variables;

%% Load data set

%Data set used for estimation (1 = pre great moderation, 2 = great moderation, 3 = pre great recession)
data_set_identifier = 3;

%Output gap measure used in data set (1 = CBO estimate, 2 = HP filter)
output_gap_identifier = 1;

if data_set_identifier == 1
    load('snk_data_pre_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    else
    data = [GAP_HP,INFL,INT];
    end
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    else
    data = [GAP_HP,INFL,INT];
    end
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    else
    data = [GAP_HP,INFL,INT];
    end
end

% Number of observations in the data set
T = size(data,1);

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Get the trend component of measurement equation and the Sigma_epsilon matrix
[~,~,~,~,~,~,t,Sigma_epsilon,~] = snk_re_build_state_space_matrices(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,theta);

%% Get values of the HEE solution

[hee] = snk_alhe3_estability(theta);

%% Get initial values for adaptive learning algorithm

[A,R_A_x,R_A_pi,z_A,B,R_B_x,R_B_pi,z_B,C,R_C_x,R_C_pi,z_C,~] = snk_alhe3_initialize_learning(number_state_variables,number_plms,hee,T);

%% Get adaptive learning state-space matrices with the initial values of the parameters

[J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2] = snk_alhe3_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_observed_variables,theta,A(:,:,1),B(:,:,1),C(:,:,1));

%% Kalman filter

%Storage for state variables
x = zeros(T,1);
pi = zeros(T,1);
u_x = zeros(T,1);
u_pi = zeros(T,1);

%Storage for expectation equations
E_A_x_tp1_all = zeros(T,1);
E_A_pi_tp1_all = zeros(T,1);
E_B_x_tp1_all = zeros(T,1);
E_B_pi_tp1_all = zeros(T,1);
E_C_x_tp1_all = zeros(T,1);
E_C_pi_tp1_all = zeros(T,1);
E_x_tp1_all = zeros(T,1);
E_pi_tp1_all = zeros(T,1);

%Storage for squared errors
E_A_x_squared_error_all = zeros(T,1);
E_A_pi_squared_error_all = zeros(T,1);
E_B_x_squared_error_all = zeros(T,1);
E_B_pi_squared_error_all = zeros(T,1);
E_C_x_squared_error_all = zeros(T,1);
E_C_pi_squared_error_all = zeros(T,1);

% Get initial values of Kalman filter
[s_bar_initial,P_initial,log_likelihood_values] = kalman_filter_initialize(n,J_1,J_2,Sigma_epsilon,T);

% Start Kalman filter algorithm
for index_1 = first_observation:T
        [s_bar,P,lik,~] = kalman_filter(data(index_1,:)',s_bar_initial,P_initial,J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2,t,Sigma_epsilon,Sigma_u_sd);
                 
        %Get values of state variables one period back
        x_tm1 = s_bar_initial(1); %Output gap
        pi_tm1 = s_bar_initial(2); %Inflation
        u_x_tm1 = s_bar_initial(4); %Aggregate demand shock
        u_pi_tm1 = s_bar_initial(5); %Aggregate supply shock
        
        %Get current values of state variables
        x_t = s_bar(1); %Output gap
        pi_t = s_bar(2); %Inflation
        u_x_t = s_bar(4); %Aggregate demand shock
        u_pi_t = s_bar(5); %Aggregate supply shock
        
        %Store current values of state variables
        x(index_1,1) = x_t; %Output gap
        pi(index_1,1) = pi_t; %Inflation
        u_x(index_1,1) = u_x_t; %Aggregate demand shock
        u_pi(index_1,1) = u_pi_t; %Aggregate supply shock
        
        %Get new regressors
        %Agent-type A
        z_A_tm1 = [u_x_tm1;u_pi_tm1];
        %Agent-type B
        z_B_tm1 = u_x_tm1;
        %Agent-type C
        z_C_tm1 = u_pi_tm1;
        
        %Get previous values of beliefs
        %Agent-type A
        A_01_tm1 = A(1,:,index_1-1)'; %Output gap equation coefficients
        A_02_tm1 = A(2,:,index_1-1)'; %Inflation equation coefficients
        %Agent-type B
        B_01_tm1 = B(1,:,index_1-1); %Output gap equation coefficients
        B_02_tm1 = B(2,:,index_1-1); %Inflation equation coefficients
        %Agent-type C
        C_01_tm1 = C(1,:,index_1-1); %Output gap equation coefficients
        C_02_tm1 = C(2,:,index_1-1); %Inflation equation coefficients
        
        %Get previous values of moment matrices
        %Agent-type A
        R_A_x_tm1 = R_A_x(:,:,index_1-1); %Output gap
        R_A_pi_tm1 = R_A_pi(:,:,index_1-1); %Inflation
        %Agent-type B
        R_B_x_tm1 = R_B_x(:,:,index_1-1); %Output gap
        R_B_pi_tm1 = R_B_pi(:,:,index_1-1); %Inflation
        %Agent-type C
        R_C_x_tm1 = R_C_x(:,:,index_1-1); %Output gap
        R_C_pi_tm1 = R_C_pi(:,:,index_1-1); %Inflation

        %Update beliefs
        %Agent-type A
        %Output gap equation coefficients
        [A_01_t,R_A_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_A,A_01_tm1,R_A_x_tm1,x_tm1,z_A_tm1); 
        %Inflation equation coefficients
        [A_02_t,R_A_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_A,A_02_tm1,R_A_pi_tm1,pi_tm1,z_A_tm1);
        
        %Agent-type B
        %Output gap equation coefficients
        [B_01_t,R_B_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_B,B_01_tm1,R_B_x_tm1,x_tm1,z_B_tm1);
        %Inflation equation coefficients
        [B_02_t,R_B_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_B,B_02_tm1,R_B_pi_tm1,pi_tm1,z_B_tm1);
        
        %Agent-type C
        %Output gap equation coefficients
        [C_01_t,R_C_x_t] = recursive_least_squares_adaptive_learning_algorithm(gn_C,C_01_tm1,R_C_x_tm1,x_tm1,z_C_tm1);
        %Inflation equation coefficients
        [C_02_t,R_C_pi_t] = recursive_least_squares_adaptive_learning_algorithm(gn_C,C_02_tm1,R_C_pi_tm1,pi_tm1,z_C_tm1);
                   
        %Store new values of beliefs
        %Agent-type A
        A(1,:,index_1) = A_01_t'; %Output gap equation coefficients
        A(2,:,index_1) = A_02_t'; %Inflation equation coefficients
        
        %Agent-type B
        B(1,:,index_1) = B_01_t; %Output gap equation coefficients
        B(2,:,index_1) = B_02_t; %Inflation equation coefficients
        
        %Agent-type C
        C(1,:,index_1) = C_01_t; %Output gap equation coefficients
        C(2,:,index_1) = C_02_t; %Inflation equation coefficients
        
        %Store regressors
        %Agent-type A
        z_A(:,index_1) = z_A_tm1;
        %Agent-type B
        z_B(:,index_1) = z_B_tm1;
        %Agent-type C
        z_C(:,index_1) = z_C_tm1;
        
        %Store new values of the moment matrices
        %Agent-type A
        R_A_x(:,:,index_1) = R_A_x_t; %Output gap
        R_A_pi(:,:,index_1) = R_A_pi_t; %Inflation
        
        %Agent-type B
        R_B_x(:,:,index_1) = R_B_x_t; %Output gap
        R_B_pi(:,:,index_1) = R_B_pi_t; %Inflation
        
        %Agent-type C
        R_C_x(:,:,index_1) = R_C_x_t; %Output gap
        R_C_pi(:,:,index_1) = R_C_pi_t; %Inflation

        %Agent-type A coefficients
        A_01_01 = A_01_t(1);
        A_01_02 = A_01_t(2);
        A_02_01 = A_02_t(1);
        A_02_02 = A_02_t(2);
        
        %Agent-type B coefficients
        B_01 = B_01_t;
        B_02 = B_02_t;
        
        %Agent-type C coefficeints
        C_01 = C_01_t;
        C_02 = C_02_t;
        
        %Agent-type A expectation equations
        %Output gap
        E_A_x_tp1 = (A_01_01*rho_x)*u_x_t + (A_01_02*rho_pi)*u_pi_t;
        E_A_x_tp1_all(index_1,1) = E_A_x_tp1;
        %Inflation
        E_A_pi_tp1 = (A_02_01*rho_x)*u_x_t + (A_02_02*rho_pi)*u_pi_t;
        E_A_pi_tp1_all(index_1,1) = E_A_pi_tp1;
        
        %Agent-type B expectation equations
        %Output gap
        E_B_x_tp1 = B_01*rho_x*u_x_t;
        E_B_x_tp1_all(index_1,1) = E_B_x_tp1;
        %Inflation
        E_B_pi_tp1 = B_02*rho_x*u_x_t;
        E_B_pi_tp1_all(index_1,1) = E_B_pi_tp1;
        
        %Agent-type C expectation equations
        %Output gap
        E_C_x_tp1 = C_01*rho_pi*u_pi_t;
        E_C_x_tp1_all(index_1,1) = E_C_x_tp1;
        %Inflation
        E_C_pi_tp1 = C_02*rho_pi*u_pi_t;
        E_C_pi_tp1_all(index_1,1) = E_C_pi_tp1;
        
        %Economy-wide expectation equations
        %Output gap
        E_x_tp1 = alpha_A*E_A_x_tp1 + alpha_B*E_B_x_tp1 + (1-alpha_A-alpha_B)*E_C_x_tp1;
        E_x_tp1_all(index_1,1) = E_x_tp1;
        %Inflation
        E_pi_tp1 = alpha_A*E_A_pi_tp1 + alpha_B*E_B_pi_tp1 + (1-alpha_A-alpha_B)*E_C_pi_tp1;
        E_pi_tp1_all(index_1,1) = E_pi_tp1;
        
        %Agent-type A squared error
        E_A_x_squared_error_all(index_1,1) = (E_A_x_tp1_all(index_1-1,1)-x_t)^2; %Output gap
        E_A_pi_squared_error_all(index_1,1) = (E_A_pi_tp1_all(index_1-1,1)-pi_t)^2; %Inflation
        
        %Agent-type B squared error
        E_B_x_squared_error_all(index_1,1) = (E_B_x_tp1_all(index_1-1,1)-x_t)^2; %Output gap
        E_B_pi_squared_error_all(index_1,1) = (E_B_pi_tp1_all(index_1-1,1)-pi_t)^2; %Inflation
        
        %Agent-type B squared error
        E_C_x_squared_error_all(index_1,1) = (E_C_x_tp1_all(index_1-1,1)-x_t)^2; %Output gap
        E_C_pi_squared_error_all(index_1,1) = (E_C_pi_tp1_all(index_1-1,1)-pi_t)^2; %Inflation
        
        %Update the state vector and the MSE matrix
        s_bar_initial = s_bar;
        P_initial = P;
        
        %Update adaptive learning state-space matrices
        [J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2] = snk_alhe3_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_observed_variables,theta,A(:,:,index_1),B(:,:,index_1),C(:,:,index_1));
end

%% Generate graphs

A_01_01_all = squeeze(A(1,1,2:end));
A_01_02_all = squeeze(A(1,2,2:end));
A_02_01_all = squeeze(A(2,1,2:end));
A_02_02_all = squeeze(A(2,2,2:end));

B_01_all = squeeze(B(1,1,2:end));
B_02_all = squeeze(B(2,1,2:end));

C_01_all = squeeze(C(1,1,2:end));
C_02_all = squeeze(C(2,1,2:end));

time = 1:T-1;

figure(1)
subplot(2,2,1)
plot(time,A_01_01_all)
xlim([0 T-1])
title('$A_{1,1}$','Interpreter','latex')

subplot(2,2,2)
plot(time,A_01_02_all)
xlim([0 T-1])
title('$A_{1,2}$','Interpreter','latex')

subplot(2,2,3)
plot(time,A_02_01_all)
xlim([0 T-1])
title('$A_{2,1}$','Interpreter','latex')

subplot(2,2,4)
plot(time,A_02_02_all)
xlim([0 T-1])
title('$A_{2,2}$','Interpreter','latex')

figure(2)
subplot(2,1,1)
plot(time,B_01_all)
xlim([0 T-1])
title('$B_1$','Interpreter','latex')

subplot(2,1,2)
plot(time,B_02_all)
xlim([0 T-1])
title('$B_2$','Interpreter','latex')

figure(3)
subplot(2,1,1)
plot(time,C_01_all)
xlim([0 T-1])
title('$C_1$','Interpreter','latex')

subplot(2,1,2)
plot(time,C_02_all)
xlim([0 T-1])
title('$C_2$','Interpreter','latex')

figure(4)
subplot(3,1,1)
plot(time,E_A_x_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^A x_{t+1}$','Interpreter','latex')

subplot(3,1,2)
plot(time,E_B_x_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^B x_{t+1}$','Interpreter','latex')

subplot(3,1,3)
plot(time,E_C_x_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^C x_{t+1}$','Interpreter','latex')

figure(5)
subplot(3,1,1)
plot(time,E_A_pi_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^A \pi_{t+1}$','Interpreter','latex')

subplot(3,1,2)
plot(time,E_B_pi_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^B \pi_{t+1}$','Interpreter','latex')

subplot(3,1,3)
plot(time,E_C_pi_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t^C \pi_{t+1}$','Interpreter','latex')

figure(6)
subplot(2,1,1)
plot(time,E_x_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t x_{t+1}$','Interpreter','latex')

subplot(2,1,2)
plot(time,E_pi_tp1_all(2:end))
xlim([0 T-1])
title('$\hat{E}_t \pi_{t+1}$','Interpreter','latex')

%% Calculate mean squared forecast errors
A_x_mean_squared_error = mean(E_A_x_squared_error_all(3:end));
A_pi_mean_squared_error = mean(E_A_pi_squared_error_all(3:end));

B_x_mean_squared_error = mean(E_B_x_squared_error_all(3:end));
B_pi_mean_squared_error = mean(E_B_pi_squared_error_all(3:end));

C_x_mean_squared_error = mean(E_C_x_squared_error_all(3:end));
C_pi_mean_squared_error = mean(E_C_pi_squared_error_all(3:end));

%% Variance of forecasts
A_x_forecast_variance = var(E_A_x_tp1_all);
A_pi_forecast_variance = var(E_A_pi_tp1_all);


B_x_forecast_variance = var(E_B_x_tp1_all);
B_pi_forecast_variance = var(E_B_pi_tp1_all);


C_x_forecast_variance = var(E_C_x_tp1_all);
C_pi_forecast_variance = var(E_C_pi_tp1_all);