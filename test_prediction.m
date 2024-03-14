% This script simulates the sampled models forward so that the resulting
% trajectories can be compared to the test data.

%% Setup
fprintf('### Testing model\n')

% Number of models
K = size(model,1);

% Simulate each model k_n times to increase robustness
k_n = 1;

% Pre-allocate
x_test_sim = zeros(n_x,T_test+1,K,k_n);
y_test_sim = zeros(T_test,n_y,K,k_n);

%% Simulate models forward
% If the Parallel Computing Toolbox is available, the for loop can be
% replaced with a parfor loop to decrease the runtime.
for kb = 1:K
    % Get model
    A = model{kb}.A;
    Q = model{kb}.Q;
    f = @(x,u) A*phi(x,u);

    for kn = 1:k_n
        % Pre-allocate
        x_loop = zeros(n_x,T_test+1);
        y_loop = zeros(T_test,n_y);

        % Get initial state
        x_loop(:,1) = model{kb}.x_prim(:,1,end);

        % Simulate model forward
        for t = 1:T_test
            x_loop(:,t+1) = f(x_loop(:,t),u_test(t)) + mvnrnd(zeros(1,n_x),Q)';
            y_loop(t,:) = g(x_loop(:,t)) + mvnrnd(zeros(1,n_y),R)';
        end
        
        % Store trajectory
        x_test_sim(:,:,kb,kn) = x_loop;
        y_test_sim(:,:,kb,kn) = y_loop;
    end
end

% Reshape
x_test_sim = reshape(x_test_sim,[n_x,T_test+1,K*k_n]);
y_test_sim = reshape(y_test_sim,[T_test,n_y,K*k_n]);

% Calculate median, mean, maximum, and minimum prediction
y_test_sim_med = median(y_test_sim,3)';
y_test_sim_mean = mean(y_test_sim,3)';

y_test_sim_max = max(y_test_sim, [], 3);
y_test_sim_min = min(y_test_sim, [], 3);

% Calculate percentiles
y_test_sim_09 = quantile(y_test_sim,0.9,3);
y_test_sim_01 = quantile(y_test_sim,0.1,3);

fprintf('### Testing complete\n')

%% Plot predictions
figure();
hold on;
plots = [];

% Plot range of predictions and percentiles
plots(end+1) = fill([t_test,flip(t_test,2)]',[y_test_sim_max; flip(y_test_sim_min,1)],0.9*[1 1 1],'linestyle','none', 'DisplayName', 'all predictions');
plots(end+1) = fill([t_test,flip(t_test,2)]',[y_test_sim_09; flip(y_test_sim_01,1)],0.7*[1 1 1],'linestyle','none', 'DisplayName', '10% perc. - 90% perc.'); 

% Plot true output
plots(end+1) = plot(t_test,y_test,'linewidth',2, 'DisplayName', 'true output'); 

% Plot median/mean prediction
% plots(end+1) = plot(t_test, y_test_sim_med, 'linewidth',2, 'DisplayName', 'median prediction');
plots(end+1) = plot(t_test, y_test_sim_mean, 'linewidth',2, 'DisplayName', 'mean prediction'); 

legend(flip(plots), 'Location','northwest')
title('Predicted Output vs. True Output')

%% Compute and print RMSE
mean_rmse = sqrt(mean((squeeze(y_test_sim)-y_test').^2,'all'));
fprintf('Mean rmse: %.2f\n', mean_rmse);
