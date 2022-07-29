function [log_prior,log_likelihood,log_posterior,solution,Phi_1,Phi_c,Phi_epsilon] = snk_re_log_posterior_calculate(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,data,theta,prior_information,Sigma_u_sd,first_observation)
%Calculates the value of the log posterior kernel of the small-scale New Keynesian model with rational
%expectations

% Input:
% number_endogenous_variables: Number of endogenous variables
% number_jumper_variables: Number of jumper variables
% number_exogenous_variables: Number of exogenous variables
% number_observed_variables: Number of observed variables
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

%Number of variables in state vector
n = number_endogenous_variables + number_jumper_variables;

%Number of observations in the data set
T = size(data,1);

% Get state space matrices and determine if the RE solution is unique and stable
[Phi_1,Phi_c,Phi_epsilon,Psi_0,Psi_1,Psi_2,t,Sigma_epsilon,solution] = snk_re_build_state_space_matrices(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,theta);

%Test for determinancy; Only calculate value of log posterior if the RE solution is unique and stable
if solution == 1 
    %Calculate value of log prior
    log_prior = log_prior_calculate(theta,prior_information);

    %Calculate value of log likelihood with the Kalman filter
    %Get initial values of Kalman filter
    [s_bar_initial,P_initial,log_likelihood_values] = kalman_filter_initialize(n,Phi_1,Phi_epsilon,Sigma_epsilon,T);

    %Start Kalman filter algorithm
    for index_1 = first_observation:T
        [s_bar,P,lik,~] = kalman_filter(data(index_1,:)',s_bar_initial,P_initial,Phi_1,Phi_c,Phi_epsilon,Psi_0,Psi_1,Psi_2,t,Sigma_epsilon,Sigma_u_sd);
        s_bar_initial = s_bar;
        P_initial = P;
        log_likelihood_values(index_1) = lik;
    end
    log_likelihood = sum(log_likelihood_values);

    % Calculate log posterior kernel
    log_posterior = log_prior + log_likelihood;
end

end

