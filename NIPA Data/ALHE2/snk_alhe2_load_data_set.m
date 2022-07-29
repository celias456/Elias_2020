function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe2_load_data_set(data_set_identifier,output_gap_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE2)

%Input
%data_set_identifier: Data set used for estimation
    %1 = pre great moderation
    %2 = great moderation
    %3 = pre great recession
%output_gap_identifier: Version of output gap used 
    %1 = CBO estimate
    %2 = HP filtered

%Parameters
% First entry is initial value (value at posterior mode - obtained from Dynare)
% Second entry is prior distribution number
% Third entry is prior distribution hyperparameter 1
% Fourth entry is prior distribution hyperparameter 2

%Output:
%data: variables in the data set
%Sigma_hat: variance of the jumping distribution used in the M-H algorithm
%c: scaling parameter used in the M-H algorithm
%first_observation: first observation used in the data set
%Sigma_u_sd: standard deviation of the measurement error term in measurement equation
%theta: vector of parameters
%prior_information: prior information for parameters in "theta" vector

if data_set_identifier == 1
    load('snk_data_pre_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.4;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.4;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe2_sigma_hat_pre_great_moderation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite

    %Parameters
    sigma_x = [0.5159,4,0.8,4.15];        
    sigma_pi = [1.2226,4,0.8,4.15];       
    sigma_i = [0.7498,4,0.8,4.15]; 
    sigma = [1.1334,2,16,0.125];         
    beta = [0.9974,1,99,1];              
    kappa = [0.9960,5,0,1];       
    i_x = [0.7865,3,0.5,0.25];            
    i_pi = [1.6139,3,1.5,0.25];       
    rho_x = [0.9575,5,0,0.97];       
    rho_pi = [0.8082,5,0,0.97]; 
    gn_A = [0.0306,5,0,0.15];
    gn_C = [0.0326,5,0,0.15];
    alpha_A = [0.6053,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.43759;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.43759;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe2_sigma_hat_great_moderation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
   
    %Parameters
    sigma_x = [0.4093,4,0.8,4.15];        
    sigma_pi = [0.7095,4,0.8,4.15];       
    sigma_i = [1.1217,4,0.8,4.15]; 
    sigma = [1.7665,2,16,0.125];         
    beta = [0.9996,1,99,1];              
    kappa = [0.9901,5,0,1];       
    i_x = [0.8037,3,0.5,0.25];            
    i_pi = [2.0710,3,1.5,0.25];       
    rho_x = [0.9691,5,0,0.97];       
    rho_pi = [0.9508,5,0,0.97]; 
    gn_A = [0.0266,5,0,0.15];
    gn_C = [0.1401,5,0,0.15];
    alpha_A = [0.7892,5,0,1];
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.48259;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.48259;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe2_sigma_hat_pre_great_recession.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.4671,4,0.8,4.15];        
    sigma_pi = [0.9947,4,0.8,4.15];       
    sigma_i = [1.0981,4,0.8,4.15]; 
    sigma = [1.2269,2,16,0.125];         
    beta = [0.9992,1,99,1];              
    kappa = [0.9939,5,0,1];       
    i_x = [0.8080,3,0.5,0.25];            
    i_pi = [1.8534,3,1.5,0.25];       
    rho_x = [0.9547,5,0,0.97];       
    rho_pi = [0.8947,5,0,0.97]; 
    gn_A = [0.0139,5,0,0.15];
    gn_C = [0.0428,5,0,0.15];
    alpha_A = [0.6926,5,0,1]; 
    
end

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_C(1);alpha_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_C(2:4);alpha_A(2:4)];
    
end

