function epsilon = epsilon(s,K,beta)
%EPSILON Determines epsilon by solving the polynomial equation.
%   Inputs:
%       s: complexity
%       K: number of scenarios (models)
%       beta: confidence parameter
%   Outputs:
%       epsilon: probability epsilon
%
%
% This function was taken from the appendix of
%
%   S. Garatti and M. C. Campi, "Risk and complexity in scenario
%   optimization", Mathematical Programming, vol. 191, no. 1, pp. 243–
%   279, 2022.
 
alphaU = 1-betaincinv(beta,K-s+1,s); 
m1 = s:1:K; 
aux1 = sum(triu(log(ones(K-s+1,1)*m1),1),2); 
aux2 = sum(triu(log(ones(K-s+1,1)*(m1-s)),1),2); 
coeffs1 = aux2-aux1;
m2 = K+1:1:4*K; 
aux3 = sum(tril(log(ones(3*K,1)*m2)),2); 
aux4 = sum(tril(log(ones(3*K,1)*(m2-s))),2); 
coeffs2 = aux3-aux4;

t1 = 0; 
t2 = 1-alphaU;
poly1 = 1+beta/(2*K)-beta/(2*K)*sum(exp(coeffs1 - (K-m1')*log(t1)))... 
    -beta/(6*K)*sum(exp(coeffs2 + (m2'-K)*log(t1)));
poly2 = 1+beta/(2*K)-beta/(2*K)*sum(exp(coeffs1 - (K-m1')*log(t2)))... 
    -beta/(6*K)*sum(exp(coeffs2 + (m2'-K)*log(t2)));
if ~((poly1*poly2) > 0)
    while t2-t1 > 1e-10 
        t = (t1+t2)/2;
        polyt =1+beta/(2*K)-beta/(2*K)*sum(exp(coeffs1-(K-m1')*log(t)))... 
            -beta/(6*K)*sum(exp(coeffs2 + (m2'-K)*log(t)));
        if polyt > 0 
            t2=t; 
        else 
            t1=t; 
        end 
    end 
    epsilon = 1-t1; 
end 
end
