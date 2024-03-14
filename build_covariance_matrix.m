function V = build_covariance_matrix(lambda, l, sf)
%BUILD_COVARIANCE_MATRIX Builds the matrix V based on a reduced-rank GP
%approximation with squared exponential kernel.
%   Inputs:
%       lambda: eigenvalues of the Laplace operator
%       l: length scale
%       sf: scale factor
%   Outputs:
%       V: left covariance matrix of MNIW prior

    V_elements = zeros(length(lambda), 1);
    for i=1:length(lambda)
        V_elements(i) = sf^2*sqrt(norm(2*pi*diag(l.^2)))*exp(-(pi^2*sqrt(lambda(i,:))*diag(l.^2)*sqrt(lambda(i,:)'))/2);
    end
    V = diag(V_elements);
end