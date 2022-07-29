function [hee] = snk_alhe3_estability(theta)
%This function determines the heterogeneous expectations equilibrium of the
%simple New Keynesian model with heterogeneous expectations (ALHE3)

%Input:
%theta: vector of parameters

%Output:
%hee.A_x_bar: HEE expressions for the coefficients on agent-type A's aggregate demand shock regressor
%hee.A_pi_bar: HEE expressions for the coefficients on agent-type A's aggregate supply shock regressor
%hee.B_bar: HEE expressions for the coefficients on agent-type B's aggregate demand shock regressor
%hee.C_bar: HEE expressions for the coefficients on agent-type A's aggregate supply shock regressor
%hee.e_stability: First element is for uniqueness, second element is for stability
    %First element equals 1 if solution is unique; 0 if not unique
    %Second element equals 1 if solution is E-stable; 0 if not E-stable

%Initialize all values 
hee.A_x_bar = 0;
hee.A_pi_bar = 0;
hee.B_bar = 0;
hee.C_bar = 0;
solution = [0,0];

%Estimated parameters
sigma = theta(4);         
beta = theta(5);       
kappa = theta(6);       
i_x = theta(7);     
i_pi = theta(8);            
rho_x = theta(9);     
rho_pi = theta(10);
alpha_A = theta(14);
alpha_B = theta(15);

%Matrices
D1 = [1,0;-kappa,1];
D2 = [alpha_A-(sigma^-1)*i_x*alpha_A,-(sigma^-1)*i_pi*alpha_A+(sigma^-1)*alpha_A;0,beta*alpha_A];
D3 = [alpha_B-(sigma^-1)*i_x*alpha_B,-(sigma^-1)*i_pi*alpha_B+(sigma^-1)*alpha_B;0,beta*alpha_B];
D4 = [(1-alpha_A-alpha_B)-(sigma^-1)*i_x*(1-alpha_A-alpha_B),-(sigma^-1)*i_pi*(1-alpha_A-alpha_B)+(sigma^-1)*(1-alpha_A-alpha_B);0,beta*(1-alpha_A-alpha_B)];
D5 = [1;0];
D6 = [0;1];

D1_inv = D1^-1;

F1 = D1_inv*D2;
F2 = D1_inv*D3;
F3 = D1_inv*D4;
F4 = D1_inv*D5;
F5 = D1_inv*D6;

n = size(D1,1);
I = eye(n);

%HEE Solution
G1 = [F1*rho_x-I,zeros(n),F2*rho_x,zeros(n);zeros(n),F1*rho_pi-I,zeros(n),F3*rho_pi;F1*rho_x,zeros(n),F2*rho_x-I,zeros(n);zeros(n),F1*rho_pi,zeros(n),F3*rho_pi-I];
G2 = [F4;F5;F4;F5];

%Test for uniqueness of HEE solution
test_1 = det(G1);

if test_1 ~= 0 %Will be true if the equilibrium is unique
    solution(1) = 1;
    Gamma_inv = G1^-1;
    HEE = -Gamma_inv*G2;

    %Test for E-stability of HEE solution
    e_stability = real(eig(G1)); %Get the real parts of the eigenvalues of the G1 matrix
    test_2 = e_stability < 0; %Will equal 1 if the real part is negative, 0 otherwise
    test_3 = sum(test_2); %Will equal 8 if all of the eigenvalues have negative real part
    
    if test_3 == 8
    solution(2) = 1; %Will equal 1 if all of the the eigenvalues have negative real part
    end

hee.A_x_bar = HEE(1:2);
hee.A_pi_bar = HEE(3:4);
hee.B_bar = HEE(5:6);
hee.C_bar = HEE(7:8);
end

hee.e_stability = solution;

end

