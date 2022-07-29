function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe3_load_data_set(data_set_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE3)

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
    c = 0.4;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_moderation_expectation.csv'); 
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
%     gn_B = [0.075,5,0,0.15];
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.33,5,0,1];
%     alpha_B = [0.33,5,0,1];
    
    %Parameters
    sigma_x = [0.7650,4,0.8,4.15];        
    sigma_pi = [0.8948,4,0.8,4.15];       
    sigma_i = [1.8685,4,0.8,4.15]; 
    sigma = [1.9818,2,16,0.125];         
    beta = [0.9971,1,99,1];              
    kappa = [0.0220,5,0,1];       
    i_x = [0.3574,3,0.5,0.25];            
    i_pi = [0.9861,3,1.5,0.25];       
    rho_x = [0.8959,5,0,0.97];       
    rho_pi = [0.9120,5,0,0.97];
    gn_A = [0.0513,5,0,0.15];
    gn_B = [0.0023,5,0,0.15];
    gn_C = [0.0670,5,0,0.15];
    alpha_A = [0.2451,5,0,1];
    alpha_B = [0.5166,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.15;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_great_moderation_expectation.csv');
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
%     gn_B = [0.075,5,0,0.15];
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.33,5,0,1];
%     alpha_B = [0.33,5,0,1];
    
    %Parameters
    sigma_x = [0.5397,4,0.8,4.15];        
    sigma_pi = [0.6445,4,0.8,4.15];       
    sigma_i = [1.5785,4,0.8,4.15]; 
    sigma = [2.2879,2,16,0.125];         
    beta = [0.9978,1,99,1];              
    kappa = [0.0007,5,0,1];       
    i_x = [0.3659,3,0.5,0.25];            
    i_pi = [1.6665,3,1.5,0.25];       
    rho_x = [0.7956,5,0,0.97];       
    rho_pi = [0.9567,5,0,0.97];
    gn_A = [0.1301,5,0,0.15];
    gn_B = [0.0059,5,0,0.15];
    gn_C = [0.0071,5,0,0.15];
    alpha_A = [0.3445,5,0,1];
    alpha_B = [0.5473,5,0,1];
    
else 
    load('snk_data_pre_great_recession_expectation.mat');

    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.2;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_recession_expectation.csv');
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
%     gn_B = [0.075,5,0,0.15];
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.33,5,0,1];
%     alpha_B = [0.33,5,0,1];

    %Parameters
    sigma_x = [0.7423,4,0.8,4.15];        
    sigma_pi = [0.5814,4,0.8,4.15];       
    sigma_i = [1.9941,4,0.8,4.15]; 
    sigma = [1.0638,2,16,0.125];         
    beta = [0.9988,1,99,1];              
    kappa = [0.0277,5,0,1];       
    i_x = [0.4081,3,0.5,0.25];            
    i_pi = [1.2972,3,1.5,0.25];       
    rho_x = [0.8697,5,0,0.97];       
    rho_pi = [0.9663,5,0,0.97];
    gn_A = [0.1260,5,0,0.15];
    gn_B = [0.0241,5,0,0.15];
    gn_C = [0.0002,5,0,0.15];
    alpha_A = [0.3222,5,0,1];
    alpha_B = [0.4912,5,0,1]; 

end

%Variables in data set
data = [EGAP,EINFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_B(1);gn_C(1);alpha_A(1);alpha_B(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_B(2:4);gn_C(2:4);alpha_A(2:4);alpha_B(2:4)];
    
end

