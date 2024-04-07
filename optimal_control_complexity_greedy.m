% This script solves the optimal control problem and determines a
% complexity s and the corresponding guarantees.

%% Setup
% Specify seed (for reproducible results)
rng(16);

% Import CasADi
addpath('') 
import casadi.*

% Confidence 1-beta = 99 %
beta = 0.01;

% Horizon
H = 100;

% Define constraints
u_min = -5;
u_max = 5;
y_min = [-Inf*ones(40,1); 2*ones(21,1); -Inf*ones(39,1)]';
y_max = Inf*ones(100,1)';

%% Set up unconstrained problem
unconstrained_problem = casadi.Opti();
U = unconstrained_problem.variable(n_u,H);
X = unconstrained_problem.variable(n_x*K,H+1);
Y = unconstrained_problem.variable(n_y*K,H);

% Cost function
unconstrained_problem.minimize(sum(U.^2));

% Initialize noise vector
v = zeros(n_x,H,K);
w = zeros(n_y,H,K);

for k = 1:K
    % Get model
    A = model{k}.A;
    Q = model{k}.Q;

    % A less efficient implementation of the basis functions phi is used 
    % here. The reason is that some of the commands used in the more
    % efficient function phi cannot be used with CasADi MX objects.
    f = @(x,u) A*phi(x,u);

    % Obtain initial state by propagation
    x_0 = f(model{k}.x_prim(:,1,end),u_training(:,end)) + mvnrnd(zeros(n_x,1),Q,1)';
    unconstrained_problem.subject_to(X(n_x*(k-1)+1:n_x*k,1) == x_0);

    % Sample noise
    v(:,:,k) = mvnrnd(zeros(1,n_x),Q,H)';
    w(:,:,k) = mvnrnd(zeros(1,n_y),R,H)';

    % Add dynamic constraints
    for t = 1:H
        unconstrained_problem.subject_to(X(n_x*(k-1)+1:n_x*k,t+1) == f(X(n_x*(k-1)+1:n_x*k,t),U(:,t)) + v(:,t,k));
        unconstrained_problem.subject_to(Y(n_y*(k-1)+1:n_y*k,t) == g(X(n_x*(k-1)+1:n_x*k,t)) + w(:,t,k));
    end
end

% Add input constraints
unconstrained_problem.subject_to(u_min <= U <= u_max);

% Set numerical backend
casadi_opts = struct('expand',1);
solver_opts = struct('linear_solver', 'ma57', 'max_iter', 1000); 
unconstrained_problem.solver('ipopt', casadi_opts, solver_opts); % Set numerical backend

%% Solve problem with all constraints
% Copy the unconstrained problem
opti = unconstrained_problem.copy();

% Add all output constraints
for k=1:K
   opti.subject_to(y_min <= Y(k,:) <= y_max);
end

fprintf('### Startet optimization of fully constrained problem.\n')
optimization_timer = tic;

% Solve OCP
sol = opti.solve();

time_optimization = toc(optimization_timer);
fprintf('### Optimization complete\nRuntime: %.2f s.\n', time_optimization)

% Extract solution
u_init = sol.value(U);
x_init = sol.value(X);
y_init = sol.value(Y);

%% Resolve the problem with initialization
% Using a proper initialization significantly reduces the runtime. Since
% the same initialization must be used to obtain guarantees, the original
% problem is solved a again. This time it is initialized with the
% solution from the first optimization.

fprintf('### Startet optimization of fully constrained problem with initialization\n')
optimization_timer = tic;

% Initialize problem
opti.set_initial(X,x_init);
opti.set_initial(U,u_init);
opti.set_initial(Y,y_init);

% Solve OCP
sol = opti.solve();

time_optimization = toc(optimization_timer); % time used for the learning
fprintf('### Optimization complete\nRuntime: %.2f s\n', time_optimization)

% Extract solution
u_opt = sol.value(U);
x_opt = sol.value(X);
y_opt = sol.value(Y);
J_opt = sol.stats.iterations.obj(end);

%% Remove constraints according to heuristic
fprintf('### Started search for support subsample\n')

% Add all models to the constraint set S
active_constraints = 1:K;

for i=1:K
    % Temporarily remove current model from the constraint set S
    temp_constraints = active_constraints(active_constraints ~= i);

    % Copy unconstrained problem
    opti = unconstrained_problem.copy();

    % Add all constraints from the current constraint set
    for k=1:length(temp_constraints)
        opti.subject_to(y_min <= Y(temp_constraints(k),:) <= y_max);
    end

    % Initialize problem
    opti.set_initial(X,x_init);
    opti.set_initial(U,u_init);
    opti.set_initial(Y,y_init);

    fprintf('Startet optimization of new constraint set\nIteration: %i/%i\n',i,K)
    optimization_timer = tic;
    
    % Solve OCP
    try
        sol = opti.solve();
    catch
        % Optimization failed, continue with next constraint set
        fprintf('Optimization failed. Continue with next constraint set\n')
        continue
    end

    time_optimization = toc(optimization_timer); % time used for the learning
    fprintf('Optimization complete\nRuntime: %.2f s\n', time_optimization)

    % Extract input trajectory
    u_opt_curr = sol.value(U);

    % Due to numerical reasons, solutions that meet the following criterion 
    % are considered equal.  
    if (max(abs(u_opt_curr - u_opt)) < 1e-6)

        % Permanently remove current model from the constraint set S
        active_constraints =  temp_constraints;
    end
end

% Get cardinality of S
s = length(active_constraints);

% Calculate guarantees
epsilon = epsilon(s,K,beta);
epsilon_perc = epsilon*100;

fprintf('### Support sub sample found\nComplexity: %i\nepsilon: %.2f %%\n',s , epsilon_perc)

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
