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
    load('snk_data_pre_great_moderation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.46987;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_moderation.csv'); 
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
    sigma_x = [0.8766,4,0.8,4.15];        
    sigma_pi = [1.2418,4,0.8,4.15];       
    sigma_i = [1.6299,4,0.8,4.15]; 
    sigma = [1.7321,2,16,0.125];         
    beta = [0.9963,1,99,1];              
    kappa = [0.9527,5,0,1];       
    i_x = [0.4072,3,0.5,0.25];            
    i_pi = [1.0057,3,1.5,0.25];       
    rho_x = [0.8915,5,0,0.97];       
    rho_pi = [0.8298,5,0,0.97];
    gn_A = [0.1432,5,0,0.15];
    gn_B = [0.1466,5,0,0.15];
    gn_C = [0.0277,5,0,0.15];
    alpha_A = [0.0182,5,0,1];
    alpha_B = [0.2816,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.25;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_great_moderation.csv');
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
    sigma_x = [0.7842,4,0.8,4.15];        
    sigma_pi = [0.7318,4,0.8,4.15];       
    sigma_i = [2.7347,4,0.8,4.15]; 
    sigma = [1.3597,2,16,0.125];         
    beta = [0.9958,1,99,1];              
    kappa = [0.9777,5,0,1];       
    i_x = [0.4614,3,0.5,0.25];            
    i_pi = [2.0038,3,1.5,0.25];       
    rho_x = [0.9528,5,0,0.97];       
    rho_pi = [0.9570,5,0,0.97];
    gn_A = [0.0638,5,0,0.15];
    gn_B = [0.0418,5,0,0.15];
    gn_C = [0.0002,5,0,0.15];
    alpha_A = [0.4011,5,0,1];
    alpha_B = [0.0005,5,0,1];
    
else 
    load('snk_data_pre_great_recession.mat');

    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.25;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_recession.csv');
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
    sigma_x = [2.3169,4,0.8,4.15];        
    sigma_pi = [1.6569,4,0.8,4.15];       
    sigma_i = [3.7007,4,0.8,4.15]; 
    sigma = [0.1872,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.9958,5,0,1];       
    i_x = [0.8147,3,0.5,0.25];            
    i_pi = [1.6813,3,1.5,0.25];       
    rho_x = [0.9488,5,0,0.97];       
    rho_pi = [0.7812,5,0,0.97];
    gn_A = [0.1453,5,0,0.15];
    gn_B = [0.0737,5,0,0.15];
    gn_C = [0.0915,5,0,0.15];
    alpha_A = [0.0014,5,0,1];
    alpha_B = [0.9799,5,0,1]; 

end

%Variables in data set
data = [GAP,INFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_B(1);gn_C(1);alpha_A(1);alpha_B(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_B(2:4);gn_C(2:4);alpha_A(2:4);alpha_B(2:4)];
    
end

