function [theta_post_burn,log_prior_post_burn,log_likelihood_post_burn,log_posterior_post_burn,acceptance_rate] = snk_alhe2_random_walk_metropolis_hastings_algorithm(theta,Sigma_hat,c,number_draws,number_endogenous_variables,number_jumper_variables,data,number_exogenous_variables,number_observed_variables,prior_information,Sigma_u_sd,burn_proportion,number_state_variables,number_plms,first_observation)
%Conducts the Metropolis-Hastings random walk algorithm for a model with
%adaptive learning and heterogeneous expectations (ALHE2)

% Inputs
% theta: Column vector of parameters
% Sigma_hat: Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
% c: Tuning parameter
% number_draws: Number of draws to generate in the algorithm
% number_endogenous_variables: Number of endogenous variables
% number_jumper_variables: Number of jumper variables
% data: Matrix of data
% number_exogenous_variables: Number of exogenous variables
% number_observed_variables: Number of observed variables
% prior_information: Matrix of prior information
% Sigma_u_sd: Standard deviation of measurement error
% burn_proportion: Proportion of draws to discard in MH chain
% number_state_variables: number of state variables
% number_plms: number of PLMs needed
% first_observation: First observation to use in the data set

%% Metropolis-Hastings algorithm characteristics

%Number of parameters
number_paramaters = size(theta,1);

mh_mu = zeros(number_paramaters,1); %Mean of the random walk term (eta) in the MH algorithm
mh_Sigma = (c^2)*Sigma_hat; %Variance of the proposal distribution in the MH algorithm

%% Create storage

%Create storage for the components of the log posterior of each draw in the MH chain
mh_theta = zeros(number_draws,number_paramaters); %Accepted draws in the MH chain
mh_theta_log_prior = zeros(number_draws,1); %Log prior of accepted draw in the MH chain
mh_theta_log_likelihood = zeros(number_draws,1); %Log likelihood of accepted draw in the MH chain
mh_theta_log_posterior_kernel = zeros(number_draws,1); %Log posterior kernel of accepted draw in the MH chain

%Create storage for the components of the log posterior of each of the candidate draws
mh_theta_candidate = zeros(number_draws,number_paramaters); %Candidate values
mh_theta_candidate_log_prior = zeros(number_draws,1); %Log prior of candidate values
mh_theta_candidate_log_likelihood = zeros(number_draws,1); %Log likelihood of candidate values
mh_theta_candidate_log_posterior_kernel = zeros(number_draws,1); %Log posterior kernel of candidate values

%Create storage for calculating acceptance rate
mh_acceptance = zeros(number_draws,1); %1 if candidate is accepted, zero otherwise

%% Initilize the Metropolis-Hastings algorithm

%Get values of components of log posterior kernel with initial parameter values

[log_prior,log_likelihood,log_posterior] = snk_alhe2_log_posterior_calculate(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,number_state_variables,number_plms,data,theta,prior_information,Sigma_u_sd,first_observation);

%Set initial values of the components of the log posterior kernel
mh_theta(1,:) = theta;
mh_theta_log_prior(1) = log_prior;
mh_theta_log_likelihood(1) = log_likelihood;
mh_theta_log_posterior_kernel(1) = log_posterior;

%% Start the Metropolis-Hastings algorithm

for mh_draws_index = 2:number_draws
    
    %Generate candidate values of parameters
    mh_theta_candidate(mh_draws_index,:) = mh_theta(mh_draws_index-1,:) + mvnrnd(mh_mu,mh_Sigma);
    
    %Get value of log posterior with candidate values
    [log_prior,log_likelihood,log_posterior] = snk_alhe2_log_posterior_calculate(number_endogenous_variables,number_jumper_variables,number_exogenous_variables,number_observed_variables,number_state_variables,number_plms,data,mh_theta_candidate(mh_draws_index,:)',prior_information,Sigma_u_sd,first_observation);
                                       
    %Test for finiteness of log posterior
    if log_posterior == -Inf %Immediately discard these candidates and move to the next step in the chain
       mh_theta(mh_draws_index,:) = mh_theta(mh_draws_index-1,:);
       mh_theta_log_prior(mh_draws_index) = mh_theta_log_prior(mh_draws_index-1);
       mh_theta_log_likelihood(mh_draws_index) = mh_theta_log_likelihood(mh_draws_index-1);
       mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_log_posterior_kernel(mh_draws_index-1);
    else
       %Value of log posterior is finite     
       mh_theta_candidate_log_prior(mh_draws_index) = log_prior;
       mh_theta_candidate_log_likelihood(mh_draws_index) = log_likelihood;
       mh_theta_candidate_log_posterior_kernel(mh_draws_index) = log_posterior;
       
       %Determine if candidate values of parameters are accepted
       mh_test = exp(mh_theta_candidate_log_posterior_kernel(mh_draws_index) - mh_theta_log_posterior_kernel(mh_draws_index-1));
       mh_alpha = min(mh_test,1);
       mh_U = unifrnd(0,1);
            if mh_U <= mh_alpha
               mh_theta(mh_draws_index,:) = mh_theta_candidate(mh_draws_index,:);
               mh_theta_log_prior(mh_draws_index) = mh_theta_candidate_log_prior(mh_draws_index);
               mh_theta_log_likelihood(mh_draws_index) = mh_theta_candidate_log_likelihood(mh_draws_index);
               mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_candidate_log_posterior_kernel(mh_draws_index);
               mh_acceptance(mh_draws_index) = 1;
            else
               mh_theta(mh_draws_index,:) = mh_theta(mh_draws_index-1,:);
               mh_theta_log_prior(mh_draws_index) = mh_theta_log_prior(mh_draws_index-1);
               mh_theta_log_likelihood(mh_draws_index) = mh_theta_log_likelihood(mh_draws_index-1);
               mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_log_posterior_kernel(mh_draws_index-1);
            end
    end
end

%Remove the burn-in draws
theta_post_burn = mh_theta((burn_proportion*number_draws)+1:end,:);
log_prior_post_burn = mh_theta_log_prior((burn_proportion*number_draws)+1:end,:);
log_likelihood_post_burn = mh_theta_log_likelihood((burn_proportion*number_draws)+1:end,:);
log_posterior_post_burn = mh_theta_log_posterior_kernel((burn_proportion*number_draws)+1:end,:);

%Calculate the acceptance rate
acceptance_rate = (sum(mh_acceptance))/number_draws;

end

