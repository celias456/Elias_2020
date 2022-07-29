function [log_prior,log_likelihood,log_posterior,solution,A,B,C] = snk_alhe1_log_posterior_calculate(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,number_state_variables,number_plms,data,theta,prior_information,Sigma_u_sd,first_observation)
%Calculates the value of the log posterior kernel for the simple New
%Keynesian model with adaptive learning and heterogeneous expectations
%(ALHE1)

% Input:
% number_endogenous_variables: Number of endogenous variables
% number_jumper_variables: Number of jumper variables
% number_exogenous_variables: Number of exogenous variables
% number_observed_variables: Number of observed variables
% number_state_variables: Number of state variables
% number_plms: Number of plms needed
% data: Matrix of data
% theta: Column vector of parameters
% prior_information: Matrix of prior information
% Sigma_u_sd: Standard deviation of measurement error
% first_observation: First observation to use in the data set

% Output:
% log_prior: value of log prior
% log_likelihood: value of log likelihood
% log_posterior: value of log posterior kernel
% solution: equals 1 if RE solution is unique and stable, 0 otherwise

%Default values for the components of the log posterior kernel
log_prior = -Inf;
log_likelihood = -Inf;
log_posterior = -Inf;

%Default values for the agent beliefs
A = 0;
B = 0;
C = 0;

%Number of variables in rational expectations state vector state vector
n = number_endogenous_variables;

%Number of observations in the data set
T = size(data,1);

%Get value of the adaptive learning gain(s)
gn_A = theta(11);
gn_B = theta(12);
gn_C = 0;

%Get the trend component of measurement equation and the Sigma_epsilon matrix, and determine if the REE solution is unique and stable
[~,~,~,~,~,~,t,Sigma_epsilon,solution] = snk_re_build_state_space_matrices(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,theta);

%Determine if the HEE solution is unique and stable
[hee] = snk_alhe1_estability(theta);
hee_test = sum(hee.e_stability);

%Only calculate value of log posterior if the REE solution is unique and stable and if the HEE solution is unique and stable
if solution == 1 && hee_test == 2
    %Calculate value of log prior
    log_prior = log_prior_calculate(theta,prior_information);
    
    %Get initial values for adaptive learning algorithm
    [A,R_A_x,R_A_pi,z_A,B,R_B_x,R_B_pi,z_B,C,R_C_x,R_C_pi,z_C,~] = snk_alhe1_initialize_learning(number_state_variables,number_plms,hee,T);
    
    %Calculate value of log likelihood with the Kalman filter
    %Get adaptive learning state-space matrices with the initial values of the parameters
    [J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2] = snk_alhe1_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_observed_variables,theta,A(:,:,1),B(:,:,1),C(:,:,1));
    %Get initial values of Kalman filter
    [s_bar_initial,P_initial,log_likelihood_values] = kalman_filter_initialize(n,J_1,J_2,Sigma_epsilon,T);

    %Start Kalman filter algorithm
    for index_1 = first_observation:T
        [s_bar,P,lik,~] = kalman_filter(data(index_1,:)',s_bar_initial,P_initial,J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2,t,Sigma_epsilon,Sigma_u_sd);
                 
        %Get values of state variables one period back
        x_tm1 = s_bar_initial(1); %Output gap
        pi_tm1 = s_bar_initial(2); %Inflation

        u_x_tm1 = s_bar_initial(4); %Aggregate demand shock
        u_pi_tm1 = s_bar_initial(5); %Aggregate supply shock
        
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

        %Record the value of the likelihood for this iteration
        log_likelihood_values(index_1) = lik;
        
        %Update the state vector and the MSE matrix
        s_bar_initial = s_bar;
        P_initial = P;
        
        %Update adaptive learning state-space matrices
        [J_1,J_c,J_2,Psi_L_0,Psi_L_1,Psi_L_2] = snk_alhe1_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_observed_variables,theta,A(:,:,index_1),B(:,:,index_1),C(:,:,index_1));
    end
    log_likelihood = sum(log_likelihood_values);

    %Calculate log posterior kernel
    log_posterior = log_prior + log_likelihood;
end

end


