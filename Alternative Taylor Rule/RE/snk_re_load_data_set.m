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
    load('snk_data_pre_great_moderation.mat');
 
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.469870; 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_pre_great_moderation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
      
    %Parameters
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
    sigma_x = [0.4291,4,0.8,4.15];        
    sigma_pi = [0.4867,4,0.8,4.15];       
    sigma_i = [1.5520,4,0.8,4.15]; 
    sigma = [1.9890,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.0001,5,0,1];       
    i_x = [0.4458,3,0.5,0.25];            
    i_pi = [0.9099,3,1.5,0.25];       
    rho_x = [0.8348,5,0,0.97];       
    rho_pi = [0.6955,5,0,0.97];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
 
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.47779;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_great_moderation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
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
    sigma_x = [0.4471,4,0.8,4.15];        
    sigma_pi = [0.8831,4,0.8,4.15];       
    sigma_i = [2.6696,4,0.8,4.15]; 
    sigma = [1.3579,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.9996,5,0,1];       
    i_x = [0.2708,3,0.5,0.25];            
    i_pi = [2.1000,3,1.5,0.25];       
    rho_x = [0.9521,5,0,0.97];       
    rho_pi = [0.9654,5,0,0.97];
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.47772;
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_re_sigma_hat_pre_great_recession.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
   
    %Parameters
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
    sigma_x = [0.4341,4,0.8,4.15];        
    sigma_pi = [1.2199,4,0.8,4.15];       
    sigma_i = [3.0682,4,0.8,4.15]; 
    sigma = [1.0425,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.9997,5,0,1];       
    i_x = [0.1186,3,0.5,0.25];            
    i_pi = [1.6674,3,1.5,0.25];       
    rho_x = [0.9390,5,0,0.97];       
    rho_pi = [0.9324,5,0,0.97];
end

%Observeable Variables
data = [GAP,INFL,INT];

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4)];
    
end

