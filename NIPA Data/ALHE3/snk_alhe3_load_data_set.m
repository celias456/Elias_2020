function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe3_load_data_set(data_set_identifier,output_gap_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE3)

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
    c = 0.2;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.3;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_moderation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.8309,4,0.8,4.15];        
    sigma_pi = [1.3419,4,0.8,4.15];       
    sigma_i = [0.7822,4,0.8,4.15]; 
    sigma = [1.3953,2,16,0.125];         
    beta = [0.9987,1,99,1];              
    kappa = [0.9260,5,0,1];       
    i_x = [0.4827,3,0.5,0.25];            
    i_pi = [1.4206,3,1.5,0.25];       
    rho_x = [0.9386,5,0,0.97];       
    rho_pi = [0.8485,5,0,0.97];
    gn_A = [0.1451,5,0,0.15];
    gn_B = [0.00001,5,0,0.15];
    gn_C = [0.0308,5,0,0.15];
    alpha_A = [0.1833,5,0,1];
    alpha_B = [0.2907,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.02;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.1;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_great_moderation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.7725,4,0.8,4.15];        
    sigma_pi = [0.9247,4,0.8,4.15];       
    sigma_i = [0.9510,4,0.8,4.15]; 
    sigma = [1.3130,2,16,0.125];         
    beta = [0.9997,1,99,1];              
    kappa = [0.9490,5,0,1];       
    i_x = [0.2328,3,0.5,0.25];            
    i_pi = [1.6264,3,1.5,0.25];       
    rho_x = [0.9571,5,0,0.97];       
    rho_pi = [0.9300,5,0,0.97];
    gn_A = [0.0684,5,0,0.15];
    gn_B = [0.00001,5,0,0.15];
    gn_C = [0.0526,5,0,0.15];
    alpha_A = [0.3279,5,0,1];
    alpha_B = [0.2994,5,0,1];
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.32;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.15;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe3_sigma_hat_pre_great_recession.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite

    %Parameters
    sigma_x = [0.4794,4,0.8,4.15];        
    sigma_pi = [0.9941,4,0.8,4.15];       
    sigma_i = [1.0975,4,0.8,4.15]; 
    sigma = [1.2168,2,16,0.125];         
    beta = [0.9973,1,99,1];              
    kappa = [0.9897,5,0,1];       
    i_x = [0.7798,3,0.5,0.25];            
    i_pi = [1.8089,3,1.5,0.25];       
    rho_x = [0.9622,5,0,0.97];       
    rho_pi = [0.8902,5,0,0.97];
    gn_A = [0.0133,5,0,0.15];
    gn_B = [0.0457,5,0,0.15];
    gn_C = [0.0428,5,0,0.15];
    alpha_A = [0.6569,5,0,1];
    alpha_B = [0.0096,5,0,1]; 

end

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_B(1);gn_C(1);alpha_A(1);alpha_B(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_B(2:4);gn_C(2:4);alpha_A(2:4);alpha_B(2:4)];
    
end

