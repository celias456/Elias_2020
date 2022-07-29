function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_alhe1_load_data_set(data_set_identifier,output_gap_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (ALHE1)

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
    c = 0.47931;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.47931;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_pre_great_moderation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.4215,4,0.8,4.15];        
    sigma_pi = [0.7440,4,0.8,4.15];       
    sigma_i = [1.2254,4,0.8,4.15]; 
    sigma = [1.6712,2,16,0.125];         
    beta = [0.9929,1,99,1];              
    kappa = [0.0006,5,0,1];       
    i_x = [0.4989,3,0.5,0.25];            
    i_pi = [1.4985,3,1.5,0.25];       
    rho_x = [0.8517,5,0,0.97];       
    rho_pi = [0.8262,5,0,0.97]; 
    gn_A = [0.1251,5,0,0.15];
    gn_B = [0.0064,5,0,0.15];
    alpha_A = [0.6162,5,0,1];
    
elseif data_set_identifier == 2
    load('snk_data_great_moderation.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.3;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.3;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_great_moderation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.5102,4,0.8,4.15];        
    sigma_pi = [0.9031,4,0.8,4.15];       
    sigma_i = [0.9033,4,0.8,4.15]; 
    sigma = [1.2137,2,16,0.125];         
    beta = [0.9988,1,99,1];              
    kappa = [0.9739,5,0,1];       
    i_x = [0.2469,3,0.5,0.25];            
    i_pi = [2.2408,3,1.5,0.25];       
    rho_x = [0.9571,5,0,0.97];       
    rho_pi = [0.9454,5,0,0.97]; 
    gn_A = [0.0004,5,0,0.15];
    gn_B = [0.0176,5,0,0.15];
    alpha_A = [0.9990,5,0,1];
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.2;
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.05;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_alhe1_sigma_hat_pre_great_recession.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.7207,4,0.8,4.15];        
    sigma_pi = [0.8818,4,0.8,4.15];       
    sigma_i = [1.7375,4,0.8,4.15]; 
    sigma = [1.0355,2,16,0.125];         
    beta = [0.9984,1,99,1];              
    kappa = [0.0005,5,0,1];       
    i_x = [0.5887,3,0.5,0.25];            
    i_pi = [2.1321,3,1.5,0.25];       
    rho_x = [0.9047,5,0,0.97];       
    rho_pi = [0.8448,5,0,0.97]; 
    gn_A = [0.0071,5,0,0.15];
    gn_B = [0.0030,5,0,0.15];
    alpha_A = [0.3401,5,0,1];

end

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1);gn_B(1);alpha_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4);gn_B(2:4);alpha_A(2:4)];
    
end

