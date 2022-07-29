function conv_results=gewconv_diagnostic(draws)
% Geweke (1992) proposed a convergence diagnostic for Markov chains based on a test for equality of the means of the first and last part of a Markov chain (by default the first 10% and the last 50%). If the samples are drawn from the stationary distribution of the chain, the two means are equal and Geweke's statistic has an asymptotically standard normal distribution.
% The test statistic is a standard Z-score: the difference between the two sample means divided by its estimated standard error. The standard error is estimated from the spectral density at zero and so takes into account any autocorrelation.
% The Z-score is calculated under the assumption that the two parts of the chain are asymptotically independent, which requires that the sum of frac1 and frac2 be strictly less than 1.

%Dynare default for frac_s1 and frac_s2 is 0.2 and 0.5, respectively (see
%page 79 of Dynare reference manual
frac_s1   = 0.2;
frac_s2   = 0.5;
[ndraws, npara]=size(draws);
% Geweke's relative numerical efficiency measures
rne_draws = momentg(draws);
% Inefficiency factors (4% tapered window)
if_draws  = zeros(npara,1);
for i=1:npara
    if_draws(i) = 1/rne_draws(i).rne1;
end

% Compute Geweke's convergence diagnostics
rne_draws1   = momentg(draws(1:frac_s1*ndraws,:));
rne_draws2   = momentg(draws(1+(frac_s2)*ndraws:end,:));
cd_draws     = apm(rne_draws1,rne_draws2);
pval_draws  = zeros(npara,4);
for i=1:npara
    for j=1:4
    pval_draws(i,j) = cd_draws(i).prob(j);
    end
end
conv_results.if = if_draws;
conv_results.pval=pval_draws;

