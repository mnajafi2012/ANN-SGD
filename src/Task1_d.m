function Task1_d
% MP2 Task 1. (d)
% run this code by simply typing Task1_d in the workspace.

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
% = w' X' X w - 2 X' t w + t' t = X' X w w - 2 X' t w + t' t =
% = 2 X' X w - 2 X' t w + t' t = 2 X' (X w - t
% The gradient of SSE is
% 2 X' X - 2 X' t = 2 X' (X - t)

% The objective is to compare the gradient of the SSE w.r.t. w for w_TRUE
% with the approx. gradient value of the same function using two different
% approximation approaches: 1.backward-difference and 2. central-difference

% 1. delta f(x) ~ (f (x) - f(x-epsilon))/epsilon; where epsilon = 10e-4
% 2. delta f(x) ~ (f (x + e) - f(x - e))/ e; where e = 10-4

% Author: Maryam Najafi
% Created date = Apr 9, 2016

close all
clear all
clc

%% 1. Generate an N-sample, i.i.d. training set

N = 100;
D = 2;

% set the seed for random numbers generation
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

t = w_TRUE' * X(1, :)' + e(1); % initialization
for i = 2 : N
    t = [t; w_TRUE' * X(i, :)' + e(i)];
end

%% 2. Generate 10 different weight vectors for the experiment
N = 10;
W = [];
W = [W; 3 * rand(N, 2)]; % random numbers between 0 and 3

%% 3. Calculate the gradient of the SSE w.r.t. w for w_TRUE
Batch_g = get_gradient(W, X, t);

%%
%% 4. Calculate the gradient of the SSE using the backward-difference approximation
epsilon = 10e-4;
Approx_g = approx_backward(epsilon, W, X, t);

%% 5. Calculate the L_2 Norm of the difference between Batch_g and Approx_g for
disp('the L_2 norm of the SSE - approx. gradient using the backward-difference approximation method');
L2_norm_backward = [];
for i = 1: N
    L2_norm_backward = [L2_norm_backward (norm (Batch_g(i)- Approx_g(i)))];
end

%%
%% 6. Calculate the gradient of the SSE using the central-difference approximation
epsilon = 10e-4;
Approx_g = approx_central(epsilon, W, X, t);

%% 7. Calculate the L_2 Norm of the difference between Batch_g and Approx_g for
disp('the L_2 norm of the SSE - approx. gradient using the central-difference approximation method');
L2_norm_central = [];
for i = 1: N
    L2_norm_central = [L2_norm_central (norm (Batch_g(i)- Approx_g(i)))];
end

%%
%% 8. Calculate the gradient of the SSE using the forward-difference approximation
epsilon = 10e-4;
Approx_g = approx_forward(epsilon, W, X, t);

%% 9. Calculate the L_2 Norm of the difference between Batch_g and Approx_g for
disp('the L_2 norm of the SSE - approx. gradient using the central-difference approximation method');
L2_norm_forward = [];
for i = 1: N
    L2_norm_forward = [L2_norm_forward (norm (Batch_g(i)- Approx_g(i)))];
end
end

function Batch_g = get_gradient (W, X, t)
% returns a matrix of 10 true gradients of the SSE from plugging in 10
% different w's.

%l =  (norm (X * w_TRUE*2 -t))^2; % the SSE

Batch_g = [];
for i = 1: size(W,1)
    batch_g = -2 * X' * ((X * W(i, :)') - t); % the gradient of the loss function (SSE)
    
    Batch_g = [Batch_g batch_g];
end

end

function Approx_g = approx_backward(epsilon, W, X, t)
% returns a matrix of 10 approximated gradients using the
% backward-difference approximation

Approx_g = []
for i = 1: size(W,1)
    f_w =  (norm (X * W(i, :)' -t))^2;
    f_w_e = (norm (X * (W(i, :)' - epsilon) -t))^2;
    
    approx_g = ( f_w - f_w_e ) / epsilon; % delta f(x) = (f(x) - f(x - epsilon)) / epsilon
    
    Approx_g = [Approx_g approx_g];
end

end

function Approx_g = approx_central (epsilon, W, X, t)
% returns a matrix of 10 approximated gradients using the
% central-difference approximation

Approx_g = []
for i = 1: size(W,1)
    f_w =  (norm (X * (W(i, :)' + epsilon) -t))^2;
    f_w_e = (norm (X * (W(i, :)' - epsilon) -t))^2;
    
    approx_g = ( f_w - f_w_e ) / 2 * epsilon; % delta f(x) = (f(x + epsilon) - f(x - epsilon)) / epsilon
    
    Approx_g = [Approx_g approx_g];
end

end

function Approx_g = approx_forward (epsilon, W, X, t)
% returns a matrix of 10 approximated gradients using the
% central-difference approximation

Approx_g = []
for i = 1: size(W,1)
    f_w =  (norm (X * (W(i, :)' + epsilon) -t))^2;
    f_w_e = (norm (X * (W(i, :)') -t))^2;
    
    approx_g = ( f_w - f_w_e ) / epsilon; % delta f(x) = (f(x + epsilon) - f(x - epsilon)) / epsilon
    
    Approx_g = [Approx_g approx_g];
end

end