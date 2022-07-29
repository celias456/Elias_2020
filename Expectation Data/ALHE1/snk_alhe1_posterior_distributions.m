function [] = snk_alhe1_posterior_distributions(theta)
%Plots the posterior distributions for the simple New Keynesian model
%with adaptive learning and heterogeneous expectations (ALHE1)

subplot(4,4,1), hist(theta(:,1));%Histogram of the posterior dist. of sigma_x
title('Posterior of $\sigma_x$','interpreter','latex')
subplot(4,4,2), hist(theta(:,2));%Histogram of the posterior dist. of sigma_pi
title('Posterior of $\sigma_{\pi}$','interpreter','latex')
subplot(4,4,3), hist(theta(:,3));%Histogram of the posterior dist. of sigma_i
title('Posterior of $\sigma_i$','interpreter','latex')
subplot(4,4,4), hist(theta(:,4));%Histogram of the posterior dist. of sigma
title('Posterior of $\sigma$','interpreter','latex')
subplot(4,4,5), hist(theta(:,5));%Histogram of the posterior dist. of beta
title('Posterior of $\beta$','interpreter','latex')
subplot(4,4,6), hist(theta(:,6));%Histogram of the posterior dist. of kappa
title('Posterior of $\kappa$','interpreter','latex')
subplot(4,4,7), hist(theta(:,7));%Histogram of the posterior dist. of i_x
title('Posterior of $i_x$','interpreter','latex')
subplot(4,4,8), hist(theta(:,8));%Histogram of the posterior dist. of i_pi
title('Posterior of $i_{\pi}$','interpreter','latex')
subplot(4,4,9), hist(theta(:,9));%Histogram of the posterior dist. of rho_x
title('Posterior of $\rho_x$','interpreter','latex')
subplot(4,4,10), hist(theta(:,10));%Histogram of the posterior dist. of rho_pi
title('Posterior of $\rho_{\pi}$','interpreter','latex')
subplot(4,4,11), hist(theta(:,11));%Histogram of the posterior dist. of gn_A
title('Posterior of $gn_A$','interpreter','latex')
subplot(4,4,12), hist(theta(:,12));%Histogram of the posterior dist. of gn_B
title('Posterior of $gn_B$','interpreter','latex')
subplot(4,4,13), hist(theta(:,13));%Histogram of the posterior dist. of alpha_A
title('Posterior of $\alpha_A$','interpreter','latex')
end

