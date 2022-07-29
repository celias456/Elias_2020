function [] = snk_alhe2_trace_plots(theta,log_posterior)
%Produces trace plots for the log posterior and each parameter of the
%simple New Keynesian model with adaptive learning and heterogeneous
%expectations (ALHE2)

n = size(theta,1);
x = (1:n);

subplot(4,4,1), plot(x,log_posterior);%Plot of draws of log posterior
title('Draws of log posterior','interpreter','latex')
subplot(4,4,2), plot(x,theta(:,1));%Plot of draws of sigma_x
title('Draws of $\sigma_x$','interpreter','latex')
subplot(4,4,3), plot(x,theta(:,2));%Plot of draws of sigma_pi
title('Draws of $\sigma_{\pi}$','interpreter','latex')
subplot(4,4,4), plot(x,theta(:,3));%Plot of draws of sigma_i
title('Draws of $\sigma_i$','interpreter','latex')
subplot(4,4,5), plot(x,theta(:,4));%Plot of draws of sigma
title('Draws of $\sigma$','interpreter','latex')
subplot(4,4,6), plot(x,theta(:,5));%Plot of draws of beta
title('Draws of $\beta$','interpreter','latex')
subplot(4,4,7), plot(x,theta(:,6));%Plot of draws of kappa
title('Draws of $\kappa$','interpreter','latex')
subplot(4,4,8), plot(x,theta(:,7));%Plot of draws of i_x
title('Draws of $i_x$','interpreter','latex')
subplot(4,4,9), plot(x,theta(:,8));%Plot of draws of i_pi
title('Draws of $i_{\pi}$','interpreter','latex')
subplot(4,4,10), plot(x,theta(:,9));%Plot of draws of rho_x
title('Draws of $\rho_x$','interpreter','latex')
subplot(4,4,11), plot(x,theta(:,10));%Plot of draws of rho_pi
title('Draws of $\rho_{\pi}$','interpreter','latex')
subplot(4,4,12), plot(x,theta(:,11));%Plot of draws of gn_A
title('Draws of $gn_A$','interpreter','latex')
subplot(4,4,13), plot(x,theta(:,12));%Plot of draws of gn_B
title('Draws of $gn_C$','interpreter','latex')
subplot(4,4,14), plot(x,theta(:,13));%Plot of draws of alpha_A
title('Draws of $\alpha_A$','interpreter','latex')
end

