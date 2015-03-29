clear all; hold off;
%% Training set: sampling 41 points in the range of [-1,1]
x_train = -1:0.05:1;
N_train = size(x_train,2); % no. of training data.
%rng(1); % seeding the randn() function.
n = randn(1,N_train);
y_train = 1.2 * sin(x_train.*pi) - cos(2.4*x_train.*pi) + 0.3*n;
%% Test set:
x_test = -1:0.01:1;
N_test=size(x_test,2);
y_test = 1.2 * sin(x_test.*pi) - cos(2.4*x_test.*pi);
%% RBFN training:
% Compute interpolation matrix.
interpolation_mat=zeros(N_train,N_train); %init
for r = 1: N_train
    for c = 1: N_train
        radius_rc=norm(x_train(1,r)-x_train(1,c),2);
        interpolation_mat(r,c)=gaussmf(radius_rc,[0.1,0]);
    end
end
% Solve for weights.
w=interpolation_mat\(y_train');
%% RBFN Test:
y_test_outcome=zeros(1,N_test); %init
for i = 1 : N_test
    y=0;%init
    for j = 1: N_train
        radius=norm(x_test(1,i) - x_train(1,j), 2);
        y = y + w(j,1) * gaussmf(radius,[0.1,0]);
    end
    y_test_outcome(1,i)=y;
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
% fprintf('Out of %d test data, %d were correctly predicted to %f precision. Accuracy = %f%%.\n', N_test, correct, epsilon, accuracy*100.0);
fprintf('SSE = %f, MSE = %f, largest abs error = %f, largest percentage error = %f%%.\n', SSE, MSE, largest_abs_error, max_frac_error*100.0);
    
