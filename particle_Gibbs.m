function model = particle_Gibbs(u_training, y_training, phi, V, g, R, x_init_mean, x_init_var, N, K, K_b, k_d, A_init, Lambda_Q, ell_Q, Q_init)
%PARTICLE_GIBBS Run particle Gibbs sampler to obtain samples from the
%joint parameter and state posterior distribution.
%   Inputs:
%       u_training: training input trajectory
%       y_training: training output trajectory
%       phi: basis functions
%       V: left covariance matrix
%       g: observation function
%       R: measurement noise covariance
%       x_init_mean: mean of the initial state
%       x_init_var: variance of the initial state
%       N: number of particles
%       K: number of models to be sampled
%       K_b: length of the burn in period
%       k_d: number of models to be skipped to decrease correlation (thinning)
%       A_init: initial value of A
%       Lambda_Q: scale matrix of IW prior on Q
%       ell_Q: degrees of freedom of IW prior on Q
%       Q_init: initial value for Q
%   Outputs:
%       model: cell array containing the different models
%
%
% This function is based on
%
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for
%   Particle Gibbs", Proceedings of the 2012 Conference on Neural
%   Information Processing Systems, Lake Taho, USA, 2012.
% 
%   A. Svensson and T. B. Schön, "A flexible state–space model for
%   learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189–
%   199, 2017.
%
% and the code provided in the supplementary material of these papers.

%% Pre-allocation and initialization
model = cell(K,1);

model{1}.A = A_init;
model{1}.Q = Q_init;

% Total number 
K_total = K_b+1+(K-1)*(k_d+1);

% Get number of states and training trajectory length
n_x = size(A_init,1);
T = size(y_training,2);

%% Particle Gibbs Sampling
learning_timer = tic; % start timer

fprintf('### Started learning algorithm\n')

x_prim = zeros(n_x,1,T); % prespecified trajectory for first iteration

for k = 1:K_total
    % Get current model
    A = model{k}.A;
    Q = model{k}.Q;
    f = @(x,u) A*phi(x,u);
    
    % Pre-allocate
    w = zeros(T,N);
    x_pf = zeros(n_x,N,T);
    a = zeros(T,N);

    % Initialize particle filter with ancestor sampling
    if k > 1
        x_pf(:,end,:) = x_prim; 
    end
    
    % Sample initial states
    for n=1:N-1
        x_pf(:,n,1) = normrnd(x_init_mean, x_init_var);
    end
    
    % Particle filter
    for t = 1:T
        % PF time propagation, resampling, and ancestor sampling
        if t >= 2
            if k > 1 % Run the conditional particle filter with ancestor sampling
                % Sample ancestors
                a(t,1:N-1) = systematic_resampling(w(t-1,:),N-1); 

                % Time propagation
                x_pf(:,1:N-1,t) = f(x_pf(:,a(t,1:N-1),t-1),repmat(u_training(t-1),1,N-1)) + mvnrnd(zeros(n_x,1),Q,N-1)'; 
                
                % Sample ancestors of prespecified trajectory
                waN = w(t-1,:).*mvnpdf(f(x_pf(:,:,t-1),repmat(u_training(t-1),1,N))',x_pf(:,N,t)',Q)'; 
                waN = waN./sum(waN); 
                a(t,N) = systematic_resampling(waN,1);

            else % Run a standard PF on first iteration
                a(t,:) = systematic_resampling(w(t-1,:),N);
                x_pf(:,:,t) = f(x_pf(:,a(t,:),t-1),repmat(u_training(t-1),1,N)) + mvnrnd(zeros(n_x,1),Q,N)' ;
            end
        end

        % PF weight update based on measurement model (logarithms are used for numerical reasons)
        log_w = -(g(x_pf(:,:,t)) - y_training(t)).^2/2/R;
        w(t,:) = exp(log_w - max(log_w));
        w(t,:) = w(t,:)/sum(w(t,:));
    end

    % Sample trajectory to condition on
    star = systematic_resampling(w(end,:),1);
    x_prim(:,1,T) = x_pf(:,star,T);
    for t = T-1:-1:1
        star = a(t+1,star);
        x_prim(:,1,t) = x_pf(:,star,t);
    end
    model{k}.x_prim = x_prim;
    
    % Sample new model parameters conditional on sampled trajectory
    zeta = squeeze(x_prim(:,1,2:T));
    z = phi(squeeze(x_prim(:,1,1:T-1)),u_training(1:T-1));
    Phi = zeta*zeta'; 
    Psi = zeta*z'; 
    Sig = z*z';
    model{k+1} = gibbs_param(Phi, Psi, Sig, V, Lambda_Q, ell_Q, T-1);

    % Print progress
    fprintf('Iteration %i/%i\n', k, K_total);
end

time_learning = toc(learning_timer); % time used for the learning
fprintf('### Learning complete\nRuntime: %.2f s\n', time_learning)

% Remove burn-in period and perform thinning
model_indices = K_b+1:k_d+1:K_total;
model = model(model_indices);
end