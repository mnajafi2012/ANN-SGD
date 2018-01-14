function Task1_c
% MP2 Task 1. (c)
% run this code by simply typing Task1_c in the workspace.

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

% The objective is to implement the SGM(Stochastic Gradient Method)


% Author: Maryam Najafi
% Created date : Apr 7, 2016
% Last modified date : Apr 9, 2016

close all
clear all
clc

%% 1. Generate an N-sample, i.i.d. training set

global w_TRUE
global D

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

%% 2. Calculate the gradient of the loss function w.r.t. w for w_TRUE
global batch_gradient

l =  (norm (X * w_TRUE*2 -t))^2; % the SSE
batch_gradient = -2 * X' * ((X * w_TRUE) - t); % the gradient of the loss function (SSE)

%% 2. Gradient Descent Method (GDM)
global upper_limit; % The termination condition based on the problem requirement

upper_limit = 10e-4;

epoch = 50;
iter = 100;

%% 2_1. for the step length etha #1
etha = 0.0001;

% 2_1_1. for the initial weight w_0
w_0 = [0.1 5]';
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

%  2_1_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

%  2_1_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);


%% 2_2. for the step length etha #2
etha = 0.01;

%  2_2_1. for the initial weight w_0
w_0 = [1 0.5]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

% 2_2_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

% 2_2_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

%% 2_3. for the step length etha #3
etha = 0.5;

%  2_3_1. for the initial weight w_0
w_0 = [1 0.5]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

%  2_3_2. for the initial weight w_0
w_0 = [2 1.3]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

%  2_3_3. for the initial weight w_0
w_0 = [4 0.4]'; % w_0
GD_const_learning_rate (etha, X, t, w_0, epoch, iter);

end

function [g_of_l, w] = update_w (X, i, t, w, etha)

global l % The loss function which is the same SSE

g_of_l = -2 * X(i, :)' * ((X(i, :) * w) - t(i)); % the gradient of the loss function (SSE)

update = etha * g_of_l; % update
w = w + update;

l = [l (norm (X * w -t))^2];
%l = [l ; tmp (norm (X * w - t))^2]; % the same answer.
end

%%
function GD_const_learning_rate(etha, X, t, w_0, epoch, iter)

global l % The loss function which is the same SSE
global w_TRUE
global D
global batch_gradient

rng (1);

w = w_0;
W = w'; % The first pair (row) in the W represents the assumption for w
l = []; % reset
G_of_l = [];

for j = 1 : epoch
    r = 0;
    for i = 1: iter
        %r = round((size(X,1) -1)  * rand(1) + 1); % pick only one sample randomly
        r = r + 1; % pick one sample ordered-base
        [g_of_l, w] = update_w(X, r, t, w, etha);
        W = [W; w'];
        G_of_l = [G_of_l; g_of_l']; % for the summation, gather the instantenous gradients  
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

% Plot w's
h = scatter(W(:,1), W(:,2),'marker','o');
set (h, 'MarkerFaceColor', [0 0 1]);
set (h, 'MarkerEdgeColor','none');
axis([0 6 0 6]);
hold on

% plot w_TRUE
scatter(w_TRUE(1),w_TRUE(2), 'x');
axis([0 6 0 6]);


% for i = 1: size(W,1)/2 % divided by 2 for a shorter computation time
%     
%     scatter(W(i,1),W(i,2), 'o');
%     axis([0 6 0 6]); % based on the iterations
%     drawnow
% end
legend('SSE','w_{TRUE}', 'w^k', 'location', 'northwest');

end