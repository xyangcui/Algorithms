function [theta]=fun_extremal_index_sueveges(Y,p,u)

%This function computes the extremal index for a time series Y, given
%quantile p and a threshold u


%REFERENCE%
%Süveges, Mária. 2007. Likelihood estimation of the extremal index.
%Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2


%INPUTS%
%Y: time series of observations
%p: quantile for the computation of extreme value theory
%u: threshold estimated via the quantile


%OUTPUT%
%theta: extremal index computed with the Sueveges formula

%% Compute theta
if nargin<3
    u=quantile(Y, p);
end

q=1-p;
Li=find(Y>u);
Ti=diff(Li);
Si=Ti-1;
Nc=length(find(Si>0));
N=length(Ti);

theta=(sum(q.*Si)+N+Nc - sqrt( (sum(q.*Si) +N+Nc).^2-8*Nc*sum(q.*Si)) )./(2*sum(q.*Si));
