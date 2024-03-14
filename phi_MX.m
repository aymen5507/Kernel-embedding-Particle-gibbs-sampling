function phi = phi_MX(n_basis_tot,n_z,L,jv,x,u)
%PHI_MX Alternative implementation of the basis functions phi that can be 
% used with CasADi MX objects. This function should only be used for the 
% optimization and not during learning since it is less efficient.
%   Inputs:
%       n_basis_tot: total number of basis functions
%       n_z: number of augmented states
%       L: interval lengths
%       jv: matrix containing all elements of the Cartesian product of the
%           vectors 1:n_basis_1,...,1:n_basis_n_z, where n_basis_i is the 
%           number of basis functions for augmented state dimension i.
%       x: state
%       u: input
%   Outputs:
%       phi: value of the basis functions evaluated at (x,u)

% Initialization
phi = ones(n_basis_tot,1);

% Augmented state
z = [u;x];

% Calculate value of basis functions. For further details see
%
%   A. Svensson and T. B. Schön, "A flexible state–space model for
%   learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189–
%   199, 2017.

for k = 1:n_z
    phi = phi.*(1./(sqrt(L(:,:,k)))*sin((pi*jv(:,:,k).*(z(k)+L(:,:,k)))./(2*L(:,:,k))));
end