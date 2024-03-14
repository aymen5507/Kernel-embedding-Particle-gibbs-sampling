function [model] = gibbs_param(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T)
%GIBBS_PARAM Sample new model parameters based on the current trajectory.
%   Inputs:
%       Phi: statistic, see paper below for definition
%       Psi: statistic, see paper below for definition
%       Sigma: statistic, see paper below for definition
%       V: left covariance matrix
%       Lambda_Q: scale matrix of IW prior on Q
%       ell_Q: degrees of freedom of IW prior on Q
%       T: length of the training trajectory
%   Outputs:
%       model: struct containing parameter samples
%
%
% This function was taken from the code provided in the supplementary material of
%
%   A. Svensson and T. B. Schön, "A flexible state–space model for
%   learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189–
%   199, 2017.

n_basis = size(V,1); % number of basis functions
n_x = size(Phi,1); % number of states

% Mean matrix of matrix normal prior on A, set to zero
M = zeros(n_x,n_basis); 

% Update statistics (only relevant if M is not zero)
Phibar = Phi + (M/V)*M';
Psibar = Psi +  M/V;

% Calculate the components of the MNIW distribution over model parameters
Sigbar = Sigma + eye(n_basis)/V;
cov_M = Lambda_Q+Phibar-(Psibar/Sigbar)*Psibar'; % scale matrix of IW distribution
cov_M_sym = 0.5*(cov_M + cov_M'); % to ensure matrix is symmetric

% Sample Q from IW distribution
Q = iwishrnd(cov_M_sym,ell_Q+T*n_x);

% Sample A from MN distribution
% See Wikipedia entry "Matrix normal distribution"
X = randn(n_x,n_basis);
post_mean = Psibar/Sigbar;
A = post_mean + chol(Q)*X*chol(eye(n_basis)/Sigbar);

% Return model parameters as struct
model.A = A;
model.Q = Q;
end
