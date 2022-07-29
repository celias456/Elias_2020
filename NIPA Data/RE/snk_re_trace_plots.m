function [] = snk_re_trace_plots(theta,log_posterior)
%Plots the post burn-in draws of each parameter for the simple new
%keynesian model with rational expectations

n = size(theta,1);
x = (1:n);

subplot(4,4,1), plot(x,log_posterior);%Plot of draws of log posterior
title('Draws of log posterior','interpreter','latex')
subplot(4,4,2), plot(x,theta(:,1));%Plot of draws of sigma_g
title('Draws of $\sigma_x$','interpreter','latex')
subplot(4,4,3), plot(x,theta(:,2));%Plot of draws of sigma_z
title('Draws of $\sigma_{\pi}$','interpreter','latex')
subplot(4,4,4), plot(x,theta(:,3));%Plot of draws of sigma_R
title('Draws of $\sigma_i$','interpreter','latex')
subplot(4,4,5), plot(x,theta(:,4));%Plot of draws of tau
title('Draws of $\sigma$','interpreter','latex')
subplot(4,4,6), plot(x,theta(:,5));%Plot of draws of kappa
title('Draws of $\beta$','interpreter','latex')
subplot(4,4,7), plot(x,theta(:,6));%Plot of draws of rho_R
title('Draws of $\kappa$','interpreter','latex')
subplot(4,4,8), plot(x,theta(:,7));%Plot of draws of psi_1
title('Draws of $r_x$','interpreter','latex')
subplot(4,4,9), plot(x,theta(:,8));%Plot of draws of psi_2
title('Draws of $r_{\pi}$','interpreter','latex')
subplot(4,4,10), plot(x,theta(:,9));%Plot of draws of rho_g
title('Draws of $\rho_x$','interpreter','latex')
subplot(4,4,11), plot(x,theta(:,10));%Plot of draws of rho_z
title('Draws of $\rho_{\pi}$','interpreter','latex')
end

