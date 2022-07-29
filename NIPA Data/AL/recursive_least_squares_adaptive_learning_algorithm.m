function [c_t,R_t] = recursive_least_squares_adaptive_learning_algorithm(gamma,c_tm1,R_tm1,y_t,x_t)
%Conducts the recursive least squares adaptive learning algorithm

%This algorithm comes from Evans and Honkapohja "Learning and Expectations
%in Macroeconomics" on pages 32-34, 334, and 349.

%gamma: adaptive learning gain
%c_tm1: k x 1 column vector of initial values of beliefs
%R_tm1: k x k initial value of the moment matrix
%y_t: most recent observation of the variable being determined
%x_t: k x 1 column vector of regressors

%Find the moment matrix
R_t = R_tm1 + gamma*(x_t*x_t' - R_tm1);

%Take the inverse of R
test = det(R_t);

if test == 0 
    R_t_inv = pinv(R_t);
else
    R_t_inv = R_t^(-1);
end

%Find the belief vector
c_t = c_tm1 + gamma*R_t_inv*x_t*(y_t - x_t'*c_tm1);


end

