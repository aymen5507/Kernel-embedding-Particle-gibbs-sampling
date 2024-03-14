% This script solves the optimal control problem one time with all
% constraints.

%% Setup
% Specify seed (for reproducible results)
rng(14);

% Import CasADi
addpath('C:\Users\Robert Lefringhausen\Documents\CasADi') 
import casadi.*

% Horizon
H = 100;

% Define constraints
u_min = -5;
u_max = 5;
y_min = [-Inf*ones(40,1); 2*ones(21,1); -Inf*ones(39,1)]';
y_max = Inf*ones(100,1)';

%% Set up OCP
opti = casadi.Opti();
X = opti.variable(n_x*K,H+1);
U = opti.variable(n_u,H);
Y = opti.variable(n_y*K,H);

% Cost function
opti.minimize(sum(U.^2));

% Initialize noise vector
v = zeros(n_x,H,K);
e = zeros(n_y,H,K);

for k = 1:K
    % Get model
    A = model{k}.A;
    Q = model{k}.Q;
    f = @(x,u) A*phi(x,u);

    % Obtain initial state by propagation
    x_0 = f(model{k}.x_prim(:,1,end),u_training(:,end)) + mvnrnd(zeros(n_x,1),Q,1)';
    opti.subject_to(X(n_x*(k-1)+1:n_x*k,1) == x_0);

    % Sample noise
    v(:,:,k) = mvnrnd(zeros(1,n_x),Q,H)';
    e(:,:,k) = mvnrnd(zeros(1,n_y),R,H)';

    % Add dynamic constraints
    for t = 1:H
        opti.subject_to(X(n_x*(k-1)+1:n_x*k,t+1) == f(X(n_x*(k-1)+1:n_x*k,t),U(:,t)) + v(:,t,k));
        opti.subject_to(Y(n_y*(k-1)+1:n_y*k,t) == g(X(n_x*(k-1)+1:n_x*k,t)) + e(:,t,k));
    end
end

% Add input constraints
opti.subject_to(u_min <= U <= u_max);

% Add all output constraints
for k=1:K
   opti.subject_to(y_min <= Y(k,:) <= y_max);
end

% Set numerical backend
casadi_opts = struct('expand',1);
solver_opts = struct('linear_solver', 'ma57', 'max_iter', 10000);
opti.solver('ipopt',casadi_opts, solver_opts); % Set numerical backend

fprintf('### Started optimization algorithm\n')
optimization_timer = tic;

% Solve OCP
sol = opti.solve();

time_optimization = toc(optimization_timer);
fprintf('### Optimization complete\nRuntime: %.2f s\n', time_optimization)

% Extract solution
u_opt = sol.value(U);
x_opt = sol.value(X);
y_opt = sol.value(Y);
J_opt = sol.stats.iterations.obj(end);

%% Plot solution
% Calculate median, mean, maximum, and minimum prediction
y_med = median(y_opt,1);
y_mean = mean(y_opt,1);

y_opt_max = max(y_opt, [], 1)';
y_opt_min = min(y_opt, [], 1)';

% Apply input trajectory to the actual system
y_sys = zeros(n_y,H);
x_sys = zeros(n_x,H+1);
x_sys(:,1) = x_training(:,end);
for t = 1:H
    x_sys(:,t+1) = f_true(x_sys(:,t),u_opt(t)) + mvnrnd(zeros(n_x,1),Q_true)';
    y_sys(:,t) = g_true(x_sys(:,t)) + mvnrnd(0,R_true,n_y);
end

% Time used for the plot
t_opt = 0:H-1;

% Plot
figure();
hold on;
plots = [];

% Plot range of predictions
plots(end+1) = fill([t_opt,flip(t_opt,2)]',[y_opt_max; flip(y_opt_min,1)],0.7*[1 1 1],'linestyle','none', 'DisplayName', 'predictions');

% Plot constraints
plots(end+1) = plot(t_opt,y_min, 'linewidth',2, 'DisplayName', 'constraints');

% Plot system output
plots(end+1) = plot(t_opt, y_sys, 'linewidth',2, 'DisplayName', 'system output');

% Plot median/mean prediction
% plots(end+1) = plot(t_opt, y_med, 'linewidth',2, 'DisplayName', 'median prediction');
plots(end+1) = plot(t_opt, y_mean, 'linewidth',2, 'DisplayName', 'mean prediction');

legend(flip(plots), 'Location','northwest')
title('Predicted Output vs. True Output')