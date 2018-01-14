function Task_Toy
% MP2 Task 1. (b)
% run this code by simply typing Task1_b in the workspace.

% The dataset X is generated for this project manually. Based on the required
% properties that are given by the problem.
% X is an NxD matrix; where N = 100 and D = 2.

% signle-output linear regression problem
% t = X w + e; where e is Normally distributed according to the following
% parameters:
% The mean = 0 and the variance v > 0.
% t is a column vector of size N; where N = 100.

% To generate the data w_TRUE should be [2 1]' and v = 1. Obviously, N and
% D are 100 and 2 respectively.

% The SSE is the loss function used for this task, which is
% || Xw - t ||^2 = (Xw - t) (Xw - t) = w' X' X w - 2 t X w + t't =
% = w' X' X w - 2 X' t w + t' t = X' X w w - 2 X' t w + t' t
% The gradient of SSE is
% 2 X' X - 2 X' t = 2 X' (X - t)

% The objective is to implement GDM using backtracking that uses step-halving.


% Author: Maryam Najafi
% Created Date: Apr 4, 2016
% Last modified date: Apt 8, 2016

close all
clear all
clc

%% 1. Generate an N-sample, i.i.d. training set
load toy
N = 8;
D = 6;

% define the seed for generating random numbers
rng(1);

% 1. generate X
left_boundary = -1;
right_boundary = 1;

X = (right_boundary - left_boundary).*rand(N,D) + left_boundary;

% 2. generate the noise e
mu = 0; % the mean of the noise
v = 1; % the variance of the noise

e = v^2.*randn(N,1) + mu;

% 3. compute t's
w_TRUE = [2 1]'; % t = w' * X + e

load toy

%% 2. Generate 100 different initial values of weight vector (w_0)
num_of_init = 1;

st = 0.01;
j = 0.9;
W_0 = [];
for i = 1: num_of_init
    j = j + st;
    %W_0 = [W_0; [j j - 1]];
    W_0 = [W_0; (1 * rand(D , 1) )']; % generate random numbers between 0 - 3
end

%% 3. Gradient Descent with Backtracking
% using step-halving
a = 0.5;

% The termination condition based on the problem requirement
global upper_limit;

upper_limit = 10e-4;

%% 3_1. for the etha_max that is very small
etha = 0.0001;
% GD_backtracking(etha, a, W_0, X, t);

%% 3_2. for the etha_max that is medium
etha = 0.1;
GD_backtracking(etha, a, W_0, X, t);

%% 3_3. for the etha_max that is very large (but GDM converges)
etha = 0.6;
% GD_backtracking(etha, a, W_0, X, t);

end

function [stop, w] = update_w (X, t, w, etha)
global upper_limit
global l % The loss function which is the same SSE

g_of_l = -2 * X' * (X * w - t); % the gradient of the loss function (SSE)

update = etha * g_of_l; % update
w = w + update;

tmp = abs(g_of_l); % Norm of the gradient
max_norm = max(tmp);

l = [l (norm (X * w -t))^2];

if (max_norm <= upper_limit)
    stop = 1;
else
    stop = 0;
end

%l = [l (norm (X * w -t))^2];
%l = [l ; tmp (norm (X * w - t))^2]; % the same answer.
end

function GD_backtracking(etha, a, W_0, X, t)
global l
num_of_iters = [];
num_of_BT_trials = [];
big_i = 0;

% figure();
for each_w_0 = 1: size(W_0 , 1)
    w = W_0 (each_w_0, :)';
    W = w'; % The first pair (row) in the W represents the assumption for w
    
    % reset
    i = 0;
    j = 0;
    stop = 0;
    l = [];
    
    while ~stop
        [stop, w] = update_w(X, t, w, etha);
        W = [W; w'];
        % update etha (step-halving)
        i = i + 1;
        big_i = big_i + 1;
        if (size(l,2) > 1)
            if l (end) > l (end -1)
                etha = a * etha;
                j = j + 1;
            end
        end
    end

    num_of_BT_trials = [num_of_BT_trials j];
    num_of_iters = [num_of_iters i];
    
    disp(W);
    % plot the average number of bt trials vs. the number of iterations
%     x = [mean(num_of_BT_trials)]; % three backtracking trials results
%     y = [big_i]; % for three etha_max
%     plot(y,x, '.');
%     xlabel('# of iterations'); ylabel('AVG # of BT-trials');
%     title(sprintf('AVG Number of BT Trials vs. Number of Iterations For etha_{max}'));
%     hold on
    
%     if (size (num_of_iters, 2) == 100)
%         avg_iters = mean(num_of_iters);
%         avg_BT = mean(num_of_BT_trials);
%         disp(fprintf('The average of the iterations is: %f', avg_iters));
%         disp(fprintf('The average of the Backtracking trials is: %f', avg_BT));
%     end
    
end


end