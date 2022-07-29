function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_re_load_data_set(data_set_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with rational expectations

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
    c = 0.47535; 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_pre_great_moderation_expectation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
      
%     %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.9999,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97];
    
    %Parameters
    sigma_x = [0.5061,4,0.8,4.15];        
    sigma_pi = [0.5309,4,0.8,4.15];       
    sigma_i = [2.1696,4,0.8,4.15]; 
    sigma = [1.2649,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.4842,5,0,1];       
    i_x = [0.1394,3,0.5,0.25];            
    i_pi = [1.4184,3,1.5,0.25];       
    rho_x = [0.7294,5,0,0.97];       
    rho_pi = [0.9632,5,0,0.97];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation_expectation.mat');
 
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.48779;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_great_moderation_expectation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
%     %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.9999,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97];

    %Parameters
    sigma_x = [0.4932,4,0.8,4.15];        
    sigma_pi = [0.7004,4,0.8,4.15];       
    sigma_i = [1.6668,4,0.8,4.15]; 
    sigma = [0.7740,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.7329,5,0,1];       
    i_x = [0.0001,3,0.5,0.25];            
    i_pi = [1.7969,3,1.5,0.25];       
    rho_x = [0.9695,5,0,0.97];       
    rho_pi = [0.9456,5,0,0.97];
    
else 
    load('snk_data_pre_great_recession_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.44931;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_pre_great_recession_expectation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
   
%     %Parameters
%     sigma_x = [1,4,0.8,4.15];        
%     sigma_pi = [1,4,0.8,4.15];       
%     sigma_i = [1,4,0.8,4.15]; 
%     sigma = [2,2,16,0.125];         
%     beta = [0.9999,1,99,1];              
%     kappa = [0.5,5,0,1];       
%     i_x = [0.5,3,0.5,0.25];            
%     i_pi = [1.5,3,1.5,0.25];       
%     rho_x = [0.49,5,0,0.97];       
%     rho_pi = [0.49,5,0,0.97]; 

    %Parameters
    sigma_x = [0.4641,4,0.8,4.15];        
    sigma_pi = [0.6718,4,0.8,4.15];       
    sigma_i = [2.1875,4,0.8,4.15]; 
    sigma = [0.4975,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.6506,5,0,1];       
    i_x = [0.0011,3,0.5,0.25];            
    i_pi = [1.4056,3,1.5,0.25];       
    rho_x = [0.9498,5,0,0.97];       
    rho_pi = [0.9471,5,0,0.97];
end

%Observeable Variables
data = [EGAP,EINFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4)];
    
end

