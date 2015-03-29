clear all; hold off;
%% Training set: sampling 41 points in the range of [-1,1]
x_train = -1:0.05:1;
N_train = size(x_train,2); % no. of training data.
%rng(1); % seeding the randn() function for reproducibility.
n = randn(1,N_train);
y_train = 1.2 * sin(x_train.*pi) - cos(2.4*x_train.*pi) + 0.3*n;
%% Test set:
x_test = -1:0.01:1;
N_test=size(x_test,2);
y_test = 1.2 * sin(x_test.*pi) - cos(2.4*x_test.*pi);
%% RBFN training:
M = 15; % no. of centers.
% Perform a k-means clustering with k=15 to obtain 15 clusters.
% Compute the means of these clusters to use as centers in RBF.
xy_train = horzcat(x_train', y_train');
[idx, centroids] = kmeans(xy_train, M);
% centroids is M x 2 matrix. Each row represents 1 cluster.
centers = centroids(:,1)';
% Compute d_max. d_max is the maximum distance between the selected
% centers.
d_max=0.0;
for i = 1 : M
    for j = 1: M
        d_max = max(d_max, norm(centers(1,i)-centers(1,j),2));
    end
end % d_max is computed.
% Define the radial basis function used.
% x is an input data and i is the i-th center.
rbf_i = @(x,i) exp( -M / d_max^2.0 * norm(x - centers(1,i), 2)^2.0 );
% Construct the interpolation matrix.
interpolation_mat = zeros(N_train, M);
for r = 1: N_train
    for c = 1: M
        interpolation_mat(r,c) = rbf_i(x_train(1,r), c);
    end
end
% Not done with interpolation matrix yet! Add column to accommodate bias.
interpolation_mat = horzcat(ones(N_train, 1), interpolation_mat);
% Since interpolation matrix isn't a square matrix,
% the weights can't be obtained by: inv(interpolation_mat) * y_train'.
% Instead, obtain the weights that minimize the SSE.
% The SSE is (y_train(1) - y_output(1))^2 + ... + (y_train(N_train) - y_output(N_train))^2.
% The optimal weights are:
w = (interpolation_mat' * interpolation_mat) \ interpolation_mat' * y_train';
%% RBFN Test:
bias = w(1,1); %retrieve bias for cleaner code later.
y_test_outcome = zeros(1,N_test); %init
for i = 1 : N_test
    for j = 1 : M
        y_test_outcome(1,i) = y_test_outcome(1,i) + w(j+1,1) * rbf_i( x_test(1,i), j);
    end
    y_test_outcome(1,i) = y_test_outcome(1,i) + bias;
end
%% Plot:
figure();
ezplot(@(x) 1.2 * sin(x*pi) - cos(2.4*x*pi),[-2,2]); %shows actual
hold on;
plot(x_test,y_test_outcome,'rx');
plot(x_test,y_test_outcome,'r-');
min_y = min(y_test_outcome); max_y=max(y_test_outcome);
min_y_axis = min_y-abs(0.1*min_y); max_y_axis = max_y + abs(0.1 * max_y);
axis([-2 2 min_y_axis max_y_axis]);
hold off;
%% Performance of RBFN:
abs_errors = abs(y_test_outcome - y_test);
% Count no. of correct predictions.
epsilon = 10 ^ -4;
correct=0; 
for i = 1 : N_test
    if( abs_errors(1, i) < epsilon)
        correct = correct + 1;
    end
end
% Accuracy
accuracy = correct / N_test;
% Compute SSE
SSE = abs_errors.^2 * ones(N_test, 1);
% Compute MSE
MSE = SSE / N_test;
% Largest error
[largest_abs_error, index_of_largest_error] = max(abs_errors);
% Largest fraction of error to associated actual y value.
frac_errors = abs_errors ./ abs(y_test);
max_frac_error = max(frac_errors);
%% Print performance:
fprintf('Out of %d test data, %d were correctly predicted to %f precision. Accuracy = %f%%.\n', N_test, correct, epsilon, accuracy*100.0);
fprintf('SSE = %f, MSE = %f, largest abs error = %f, largest percentage error = %f%%.\n', SSE, MSE, largest_abs_error, max_frac_error*100.0);