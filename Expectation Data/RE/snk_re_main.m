%%
%Simple New Keynesian Model with Rational Expectations
%Main file

clear
clc

%% Set seed state
seed = 123;
rng(seed);

%% Characteristics of model

number_endogenous_variables = 5; %Number of endogenous variables
number_jumper_variables = 2; %Number of jumper variables
number_exogenous_variables = 3; %Number of exogenous variables
number_observed_variables = 3; %Number of observable variables

%% Load data set

%Data set used for estimation
%1 = pre great moderation
%2 = great moderation
%3 = pre great recession
data_set_identifier = 3;

[data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_re_load_data_set(data_set_identifier);

%% Metropolis-Hastings (MH) characteristics

number_draws = 500000; %Number of MH draws
burn_proportion = 0.5; %Proportion of draws to discard in MH chain
percentile = 0.1; %Percentile for interval estimates
log_marginal_likelihood_tau = 0.95; %Tuning parameter used in the calculation of the modified harmonic mean

%% Get value of log posterior kernel with the initial parameter values
[log_prior,log_likelihood,log_posterior,solution,Phi_1,Phi_c,Phi_epsilon] = snk_re_log_posterior_calculate(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,data,theta,prior_information,Sigma_u_sd,first_observation);

%% Run Metropolis-Hastings algorithm
[theta_post_burn,log_prior_post_burn,log_likelihood_post_burn,log_posterior_post_burn,acceptance_rate] = snk_re_random_walk_metropolis_hastings_algorithm(theta,Sigma_hat,c,number_draws,number_endogenous_variables,number_jumper_variables,data,number_exogenous_variables,number_observed_variables,prior_information,Sigma_u_sd,burn_proportion,first_observation);

%% Get estimates
[estimates_point,estimates_interval] = bayesian_estimates(theta_post_burn,percentile);

%% Get value of log marginal likelihood using the modified harmonic mean
log_marginal_likelihood = modified_harmonic_mean(theta_post_burn,log_prior_post_burn,log_likelihood_post_burn,log_marginal_likelihood_tau);

%% Create posterior distribution plots
figure(1)
snk_re_posterior_distributions(theta_post_burn);

%% Create trace plots
figure(2)
snk_re_trace_plots(theta_post_burn,log_posterior_post_burn);

%% Convergence diagnostics
convergence_diagnostics = gewconv_diagnostic(theta_post_burn);