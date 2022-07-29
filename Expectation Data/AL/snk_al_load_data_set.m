function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_al_load_data_set(data_set_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (AL)

%Input
%data_set_identifier: Data set used for estimation
    %1 = pre great moderation
    %2 = great moderation
    %3 = pre great recession

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
    load('snk_data_pre_great_moderation_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.3; 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_al_sigma_hat_pre_great_moderation_expectation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.99,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97];
%     gn_A = [0.075,5,0,0.15];
   
    %Parameters
    sigma_x = [0.5796,4,0.8,4.15];        
    sigma_pi = [0.4729,4,0.8,4.15];       
    sigma_i = [1.8170,4,0.8,4.15]; 
    sigma = [1.5222,2,16,0.125];         
    beta = [0.9981,1,99,1];              
    kappa = [0.0012,5,0,1];       
    i_x = [0.4838,3,0.5,0.25];            
    i_pi = [0.9780,3,1.5,0.25];       
    rho_x = [0.8191,5,0,0.97];       
    rho_pi = [0.6628,5,0,0.97]; 
    gn_A = [0.1242,5,0,0.15];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.25; 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_al_sigma_hat_great_moderation_expectation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.99,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97];
%     gn_A = [0.075,5,0,0.15];
    
    %Parameters
    sigma_x = [0.8048,4,0.8,4.15];        
    sigma_pi = [0.4797,4,0.8,4.15];       
    sigma_i = [1.7010,4,0.8,4.15]; 
    sigma = [0.3303,2,16,0.125];         
    beta = [0.9905,1,99,1];              
    kappa = [0.4659,5,0,1];       
    i_x = [-0.0027,3,0.5,0.25];            
    i_pi = [1.6138,3,1.5,0.25];       
    rho_x = [0.9700,5,0,0.97];       
    rho_pi = [0.9285,5,0,0.97]; 
    gn_A = [0.0043,5,0,0.15]; 
    
else 
    load('snk_data_pre_great_recession_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.2; 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_al_sigma_hat_pre_great_recession_expectation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.99,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97];
%     gn_A = [0.075,5,0,0.15];
       
    %Parameters
    sigma_x = [0.5484,4,0.8,4.15];        
    sigma_pi = [0.2495,4,0.8,4.15];       
    sigma_i = [2.0052,4,0.8,4.15]; 
    sigma = [0.9264,2,16,0.125];         
    beta = [0.8750,1,99,1];              
    kappa = [0.0237,5,0,1];       
    i_x = [0.3673,3,0.5,0.25];            
    i_pi = [1.1674,3,1.5,0.25];       
    rho_x = [0.8765,5,0,0.97];       
    rho_pi = [0.8216,5,0,0.97]; 
    gn_A = [0.0492,5,0,0.15];
    
end

%Variables in data set
data = [EGAP,EINFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4)];

end

