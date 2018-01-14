function Task1_a
% MP2 Task 1. (a)
% run this code by simply typing Task1_a in the workspace.

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

% The objective is to implement the GDM that uses a constant learning rate.


% Author: Maryam Najafi
% Created date: Apr 4, 2016
% Last modified date = Apr 7, 2016

close all
clear all
clc

%% 1. Generate an N-sample, i.i.d. training set

global w_TRUE
global D

N = 100;
D = 2;

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

%% 2. Gradient Descent Method (GDM)

global upper_limit; % The termination condition based on the problem requirement

upper_limit = 10e-4;

outter_iter = 50;
inner_iter = 200;

%% 2_1. for the step length etha #1
etha = 0.0001;

% 2_1_1. for the initial weight w_0
w_0 = [0.1 5]';
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%  2_1_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%  2_1_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%% 2_2. for the step length etha #2
etha = 0.01;

%  2_2_1. for the initial weight w_0
w_0 = [1 0.5]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

% 2_2_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

% 2_2_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%% 2_3. for the step length etha #3
etha = 0.5;

%  2_3_1. for the initial weight w_0
w_0 = [1 0.5]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%  2_3_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

%  2_3_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, outter_iter, inner_iter);

end

function [stop, w] = update_w (X, t, w, etha)
global upper_limit
global l % The loss function which is the same SSE

g_of_l = -2 * X' * (X * w - t); % the gradient of the loss function (SSE)

update = etha * g_of_l; % update
w = w + update;

tmp = abs(g_of_l); % Norm of the gradient
max_norm = max(tmp);

if (max_norm < upper_limit)
    stop = 1;
else
    stop = 0; 
end

l = [l (norm (X * w -t))^2];

end

function GD_const_learning_rate(etha, X, t, w_0, outter_iter, inner_iter)

global l % The loss function which is the same SSE
global w_TRUE
global D

w = w_0;
W = w'; % The first pair (row) in the W represents the assumption for w
for j = 1 : outter_iter
    for i = 1: inner_iter
        [stop, w] = update_w(X, t, w, etha);
        W = [W; w'];
        if stop
            break
        end
    end
    if stop
        break
    end
end

% Plot SSE vs. iteration
figure();
x_axis = (1:1:size(l,2)); % based on the iterations
plot (x_axis, l, 'Color' , 'g'); % the SSE curve
xlabel('iter'); ylabel('SSE');
title('SSE vs. iteration');

figure()
% Plot the SSE contours
mu = w_TRUE'; % The bottom of the valley
sigma = cov(l) * eye(D);

step = 0.01; % the contour plot resolution

% axis range
leftboundary = 0;
rightboundary = 6;

m1 = leftboundary:step:rightboundary; m2 = leftboundary:step:rightboundary;
[M1,M2] = meshgrid(m1,m2);

F = mvnpdf([M1(:) M2(:)],mu,sigma);
F = reshape(F,length(m2),length(m1));

contour(m1,m2,F);
xlabel('w_1'); ylabel('w_2');
title(sprintf('etha(learning rate)= %6.2E, w_0= [%6.2E; %6.2E]', etha, w_0(1), w_0(2)));
hold on

scatter(w_TRUE(1),w_TRUE(2), 'x');
axis([0 6 0 6]);
hold on

% Plot w's
for i = 1: size(W,1)/2 % divided by 2 for a shorter computation time

scatter(W(i,1),W(i,2), 'o');
axis([0 6 0 6]); % based on the iterations
drawnow
end
legend('SSE','w_{TRUE}', 'w^k', 'location', 'northwest');

end