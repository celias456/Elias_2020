function [estimates_point,estimates_interval] = bayesian_estimates(mh_theta_post_burn_in,percentile)
%Calculates Bayesian point and interval estimates

%Point estimates
estimates_point(1,:) = mean(mh_theta_post_burn_in);
estimates_point(2,:) = median(mh_theta_post_burn_in);
estimates_point(3,:) = mode(mh_theta_post_burn_in);

%Interval estimates (posterior probability interval)
upper_bound = 100-0.5*(100*percentile);
lower_bound = 0.5*(100*percentile);
estimates_interval(1,:) = prctile(mh_theta_post_burn_in,lower_bound);
estimates_interval(2,:) = prctile(mh_theta_post_burn_in,upper_bound);
end

