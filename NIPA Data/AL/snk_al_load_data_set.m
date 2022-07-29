function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information] = snk_al_load_data_set(data_set_identifier,output_gap_identifier)
%Loads the data set and associated characteristics for the simple New
%Keynesian model with heterogeneous expectations (AL)

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
    Sigma_hat = load('snk_al_sigma_hat_pre_great_moderation.csv'); 
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
   
    %Parameters
    sigma_x = [0.4048,4,0.8,4.15];        
    sigma_pi = [0.5034,4,0.8,4.15];       
    sigma_i = [1.2418,4,0.8,4.15]; 
    sigma = [2.0764,2,16,0.125];         
    beta = [0.9960,1,99,1];              
    kappa = [0.0026,5,0,1];       
    i_x = [0.5098,3,0.5,0.25];            
    i_pi = [1.1728,3,1.5,0.25];       
    rho_x = [0.7803,5,0,0.97];       
    rho_pi = [0.6962,5,0,0.97]; 
    gn_A = [0.1396,5,0,0.15];
    
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
    Sigma_hat = load('snk_al_sigma_hat_great_moderation.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
    
    %Parameters
    sigma_x = [0.4258,4,0.8,4.15];        
    sigma_pi = [0.8978,4,0.8,4.15];       
    sigma_i = [1.0645,4,0.8,4.15]; 
    sigma = [1.4211,2,16,0.125];         
    beta = [0.9967,1,99,1];              
    kappa = [0.9833,5,0,1];       
    i_x = [0.2258,3,0.5,0.25];            
    i_pi = [2.1380,3,1.5,0.25];       
    rho_x = [0.9697,5,0,0.97];       
    rho_pi = [0.9459,5,0,0.97]; 
    gn_A = [0.0004,5,0,0.15]; 
    
else 
    load('snk_data_pre_great_recession.mat');
    
    %Variables in data set
    if output_gap_identifier == 1
    data = [GAP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.1; 
    else
    data = [GAP_HP,INFL,INT];
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.06;
    end
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('snk_al_sigma_hat_pre_great_recession.csv');
    Sigma_hat = (Sigma_hat + Sigma_hat')/2; %This ensures that the Sigma_hat matrix is positive semi-definite
       
    %Parameters
    sigma_x = [0.4601,4,0.8,4.15];        
    sigma_pi = [1.2910,4,0.8,4.15];       
    sigma_i = [1.1228,4,0.8,4.15]; 
    sigma = [0.9972,2,16,0.125];         
    beta = [0.9999,1,99,1];              
    kappa = [0.9995,5,0,1];       
    i_x = [0.12,3,0.5,0.25];            
    i_pi = [1.7896,3,1.5,0.25];       
    rho_x = [0.9498,5,0,0.97];       
    rho_pi = [0.9299,5,0,0.97]; 
    gn_A = [0.0001,5,0,0.15];
    
end

%First observation in the data set
first_observation = 2;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = [sigma_x(1);sigma_pi(1);sigma_i(1);sigma(1);beta(1);kappa(1);i_x(1);i_pi(1);rho_x(1);rho_pi(1);gn_A(1)];

%Stack the prior information in a matrix
prior_information = [sigma_x(2:4);sigma_pi(2:4);sigma_i(2:4);sigma(2:4);beta(2:4);kappa(2:4);i_x(2:4);i_pi(2:4);rho_x(2:4);rho_pi(2:4);gn_A(2:4)];

end

