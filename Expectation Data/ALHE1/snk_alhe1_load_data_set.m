function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe1_load_data_set(data_set_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE1)

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
    c = 0.35;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_pre_great_moderation_expectation.csv'); 
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
%     alpha_A = [0.5,5,0,1];
    
    %Parameters
    sigma_x = [0.5491,4,0.8,4.15];        
    sigma_pi = [0.6409,4,0.8,4.15];       
    sigma_i = [1.8450,4,0.8,4.15]; 
    sigma = [1.8373,2,16,0.125];         
    beta = [0.9950,1,99,1];              
    kappa = [0.0016,5,0,1];       
    i_x = [0.5314,3,0.5,0.25];            
    i_pi = [0.9101,3,1.5,0.25];       
    rho_x = [0.8509,5,0,0.97];       
    rho_pi = [0.9164,5,0,0.97]; 
    gn_A = [0.0884,5,0,0.15];
    gn_B = [0.0058,5,0,0.15];
    alpha_A = [0.5541,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.11;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_great_moderation_expectation.csv');
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
%     alpha_A = [0.5,5,0,1];
    
    %Parameters
    sigma_x = [0.4450,4,0.8,4.15];        
    sigma_pi = [0.5447,4,0.8,4.15];       
    sigma_i = [1.4821,4,0.8,4.15]; 
    sigma = [1.1660,2,16,0.125];         
    beta = [0.9967,1,99,1];              
    kappa = [0.0001,5,0,1];       
    i_x = [0.3322,3,0.5,0.25];            
    i_pi = [1.6670,3,1.5,0.25];       
    rho_x = [0.8824,5,0,0.97];       
    rho_pi = [0.9680,5,0,0.97]; 
    gn_A = [0.0343,5,0,0.15];
    gn_B = [0.0320,5,0,0.15];
    alpha_A = [0.4902,5,0,1];
    
else 
    load('snk_data_pre_great_recession_expectation.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.15;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_pre_great_recession_expectation.csv');
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
%     alpha_A = [0.5,5,0,1];
    
    %Parameters
    sigma_x = [0.4584,4,0.8,4.15];        
    sigma_pi = [0.5947,4,0.8,4.15];       
    sigma_i = [2.0469,4,0.8,4.15]; 
    sigma = [1.6650,2,16,0.125];         
    beta = [0.9968,1,99,1];              
    kappa = [0.0114,5,0,1];       
    i_x = [0.4819,3,0.5,0.25];            
    i_pi = [1.2630,3,1.5,0.25];       
    rho_x = [0.8329,5,0,0.97];       
    rho_pi = [0.9532,5,0,0.97]; 
    gn_A = [0.1170,5,0,0.15];
    gn_B = [0.0002,5,0,0.15];
    alpha_A = [0.5050,5,0,1];

end

%Variables in data set
data = [EGAP,EINFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_B(1);alpha_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_B(2:4);alpha_A(2:4)];
    
end

