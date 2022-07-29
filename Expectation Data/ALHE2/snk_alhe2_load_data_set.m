function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe2_load_data_set(data_set_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE2)

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
    Sigma_hat = load('snk_alhe2_sigma_hat_pre_great_moderation_expectation.csv'); 
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
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.5,5,0,1];

    %Parameters
    sigma_x = [0.9008,4,0.8,4.15];        
    sigma_pi = [0.4349,4,0.8,4.15];       
    sigma_i = [1.7661,4,0.8,4.15]; 
    sigma = [1.4786,2,16,0.125];         
    beta = [0.9969,1,99,1];              
    kappa = [0.0041,5,0,1];       
    i_x = [0.3954,3,0.5,0.25];            
    i_pi = [0.9265,3,1.5,0.25];       
    rho_x = [0.8166,5,0,0.97];       
    rho_pi = [0.6831,5,0,0.97]; 
    gn_A = [0.1072,5,0,0.15];
    gn_C = [0.0157,5,0,0.15];
    alpha_A = [0.8258,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.1;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe2_sigma_hat_great_moderation_expectation.csv');
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
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.5,5,0,1];
   
    %Parameters
    sigma_x = [1.2100,4,0.8,4.15];        
    sigma_pi = [0.4229,4,0.8,4.15];       
    sigma_i = [1.6291,4,0.8,4.15]; 
    sigma = [0.8800,2,16,0.125];         
    beta = [0.9987,1,99,1];              
    kappa = [0.4789,5,0,1];       
    i_x = [0.1247,3,0.5,0.25];            
    i_pi = [1.6384,3,1.5,0.25];       
    rho_x = [0.9591,5,0,0.97];       
    rho_pi = [0.9203,5,0,0.97]; 
    gn_A = [0.0840,5,0,0.15];
    gn_C = [0.0001,5,0,0.15];
    alpha_A = [0.4114,5,0,1];
    
else 
    load('snk_data_pre_great_recession_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.1;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe2_sigma_hat_pre_great_recession_expectation.csv');
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
%     gn_C = [0.075,5,0,0.15];
%     alpha_A = [0.5,5,0,1];
    
    %Parameters
    sigma_x = [1.3048,4,0.8,4.15];        
    sigma_pi = [0.3583,4,0.8,4.15];       
    sigma_i = [2.0917,4,0.8,4.15]; 
    sigma = [1.2226,2,16,0.125];         
    beta = [0.9995,1,99,1];              
    kappa = [0.4565,5,0,1];       
    i_x = [0.1765,3,0.5,0.25];            
    i_pi = [1.4707,3,1.5,0.25];       
    rho_x = [0.9418,5,0,0.97];       
    rho_pi = [0.9395,5,0,0.97]; 
    gn_A = [0.1080,5,0,0.15];
    gn_C = [0.0001,5,0,0.15];
    alpha_A = [0.3898,5,0,1]; 
    
end

%Variables in data set
data = [EGAP,EINFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_C(1);alpha_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_C(2:4);alpha_A(2:4)];
    
end

