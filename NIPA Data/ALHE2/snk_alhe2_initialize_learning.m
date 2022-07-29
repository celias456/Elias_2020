function [A,R_A_x,R_A_pi,z_A,B,R_B_x,R_B_pi,z_B,C,R_C_x,R_C_pi,z_C,state_variables] = snk_alhe2_initialize_learning(number_state_variables,number_plms,hee,T)
%Initializes the adaptive learning algorithm for the small-scale New
%Keynesian model with heterogeneous expectations (ALHE2)

%Get number of regressors
number_regressors_A = number_state_variables;
number_regressors_B = 1;
number_regressors_C = 1;

%Create storage for beliefs
A = zeros(number_plms,number_regressors_A,T); %Agent-type A beliefs
B = zeros(number_plms,number_regressors_B,T); %Agent-type B beliefs
C = zeros(number_plms,number_regressors_C,T); %Agent-type C beliefs

%Get initial values of beliefs from rational expectations solution
%Agent-type A
%Output Gap
A(1,1,1) = hee.A_x_bar(1);
A(1,2,1) = hee.A_pi_bar(1);

%Inflation
A(2,1,1) = hee.A_x_bar(2);
A(2,2,1) = hee.A_pi_bar(2);

%Agent-type B
%Output Gap
B(1,1,1) = hee.B_bar(1);

%Inflation
B(2,1,1) = hee.B_bar(2);

%Agent-type C
%Output Gap
C(1,1,1) = hee.C_bar(1);

%Inflation
C(2,1,1) = hee.C_bar(2);


%Create storage for moment matrices
%Agent-type A
R_A_x = zeros(number_regressors_A,number_regressors_A,T); %Output gap
R_A_pi= zeros(number_regressors_A,number_regressors_A,T); %Inflation

%Agent-type B
R_B_x = zeros(number_regressors_B,number_regressors_B,T); %Output gap
R_B_pi = zeros(number_regressors_B,number_regressors_B,T); %Inflation

%Agent-type C
R_C_x = zeros(number_regressors_C,number_regressors_C,T); %Output gap
R_C_pi = zeros(number_regressors_C,number_regressors_C,T); %Inflation

%Store initial values of moment matrices
%Agent-type A
R_A_x(:,:,1) = eye(number_regressors_A);
R_A_pi(:,:,1) = eye(number_regressors_A);

%Agent-type B
R_B_x(:,:,1) = eye(number_regressors_B);
R_B_pi(:,:,1) = eye(number_regressors_B);

%Agent-type C
R_C_x(:,:,1) = eye(number_regressors_C);
R_C_pi(:,:,1) = eye(number_regressors_C);

%Create storage for regressors
%Agent-type A
z_A = zeros(number_regressors_A,T);

%Agent-type B
z_B = zeros(number_regressors_B,T);

%Agent-type C
z_C = zeros(number_regressors_C,T);

%Store initial values of regressors
%Agent-type A
z_A(1,1) = 1;

%Agent-type B
z_B(1,1) = 1;

%Agent-type C
z_C(1,1) = 1;

%Create storage for state variables
state_variables = zeros(T,number_state_variables);
end

