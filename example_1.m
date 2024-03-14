%% First example
% This code reproduces the first example of the paper "Learning-Based 
% Optimal Control with Performance Guarantees for Unknown Systems with 
% Latent States", available as pre-print on arXiv, 
% https://arxiv.org/abs/2303.17963.

% Clear
clear;
clc;
close all;

% Specify seed (for reproducible results)
rng(15);

%% Model parameters
% Number of states etc.
n_x = 2; % number of states
n_u = 1; % number of control inputs
n_y = 1; % number of outputs

% Define measurement model - assumed to be known
g = @(x) [1, 0]*x; % observation function
R = 0.1; % measurement noise variance

% Normally distributed initial state
x_init_mean = [2;2]; % mean
x_init_var = 0.5*ones(n_x,1); % variance

%% Learning parameters
K = 20; % number of models to be sampled
K_b = 20; % length of burn-in period
k_d = 0; % number of models to be skipped to decrease correlation (thinning)
N = 30; % number of particles of SMC algorithm

%% Priors for Q - inverse Wishart distribution
ell_Q = 5; % degrees of freedom
Lambda_Q = eye(n_x); % scale matrix

%% Basis functions and priors
% Basis functions - assumed to be known
phi = @(x,u) [x(1,:); x(2,:); u(1,:); cos(3*x(1,:)).*x(2,:); sin(2*x(2,:)).*u(1,:)];
n_phi = 5; % number of basis functions

% Parameter of the prior for A
V = diag(2*ones(n_phi,1));

% Initial A
A_init = zeros(n_x, n_phi);

%% Parameters for data generation
T = 2000; % number of steps for training
T_test = 500;  % number of steps for testing
T_all = T + T_test;

t_data = 0:T_all-1;
t_training = 1:T;
t_test = T+1:T_all;

%% Generate data
% Choose the actual system (to be learned) and generate input-output data
% of length T.
% x_t+1 = f(x_t,u_t) + N(0,Q)
% y_t = g(x_t,u_t) + N(0,R)

% Unknown system
f_true = @(x,u) [0.8*x(1)-0.5*x(2)+0.1*cos(3*x(1))*x(2);0.4*x(1)+0.5*x(2)+(1+0.3*sin(2*x(2)))*u];
Q_true = [0.03 -0.004; -0.004, 0.01];
g_true = g;
R_true = R;

% Input trajectory
u_training = mvnrnd(0,3,T)'; % u_training = 3-6*prbs(50,T);
u_test = 3*sin(2*pi*(1/T_test)*((1:T_test)-1));
u = [u_training, u_test];

% Generate data by forward simulation
x = zeros(n_x,T_all+1);
x(:,1) = normrnd(x_init_mean, x_init_var);
y = zeros(n_y,T_all);

for t = 1:T_all
    x(:,t+1) = f_true(x(:,t),u(t)) + mvnrnd(zeros(n_x,1),Q_true)';
    y(:,t) = g_true(x(:,t)) + mvnrnd(0,R_true,n_y);
end

% Split data into training and test data
u_training = u(:,1:T);
x_training = x(:,1:T+1);
y_training = y(:,1:T);

u_test = u(:,T+1:end);
x_test = x(:,T+1:end);
y_test = y(:,T+1:end);

% % Plot data
figure()
hold on
u_plot = plot(u, 'linewidth',1);
y_plot = plot(y, 'linewidth',1);
legend([u_plot, y_plot], 'input', 'output')
uistack(u_plot,'top')
ylabel('u | y')
xlabel('t')

%% Learn models
% Result: K models of the type
% x_t+1 = model{i}.A*phi(x_t,u_t) + N(0,model{i}.Q),
% where phi are the basis functions defined above.
model = particle_Gibbs(u_training, y_training, phi, V, g, R, x_init_mean, x_init_var, N, K, K_b, k_d, A_init, Lambda_Q, ell_Q, Lambda_Q);

%% Evaluate
% Test the models with the test dataset by simulating it forward in time.
test_prediction

%% Solve OCP and plot solution
% optimal_control

%% Solve OCP and determine complexity s via greedy algorithm
% optimal_control_complexity_greedy