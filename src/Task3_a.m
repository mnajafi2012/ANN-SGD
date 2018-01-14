function Task3_a
% MP2 Task 3. (a)
% run this code by simply typing Task3_a in the workspace.

% The objective is to draw 1000 samples out of the feature space F whose 
%vectors are in the interval [-1 1] x [-1 1]. Each element x = {x_1 x_2}^T;
%x_1 and x_2 varies between -1 and 1
% label each sample in accordance with N the GXOR problem (-1 and 1) to
% generate a set of labeled data

% Author: Maryam Najafi
% Created date: Apr 11, 2016
% Last modified date: Apr 13, 2016

close all
clc
clear all

% # of samples
N = 1000;
% data dimension
d = 2;
% # of the training set samples
N_train = 50;
% # of the validation set samples
N_val = 450;
% # of the testing set samples
N_test = N - N_train + N_val;

%% 1. generate a sample matrix of N vectors (samples) D = 1000 x 2
D = generateSamples(N, d);
%% 2. generate a target vector of N scalar numbers T = 1000 x 1
T = generateTarget(D, N);
%% 3. create the training set
D_train = D(:, 1:N_train);
%% 4. create the validation set
D_val = D(:, N_train+1:N_train + N_val);
%% 5. create the testing set
D_test = D(:, N_val+1:N);
%% 6. plot two classes
plot_samples(D, T);


end

function D = generateSamples(N, d)
% D represents 1000 samples and stands for the Dataset
D = 2 * rand(d,N) - 1; % samples are uniformly distributed between -1 and 1

end

function T = generateTarget(D, N)
% target labels are either 0 or 1
T = [];
for i = 1:N
   if D(1,i) * D(2,i) >= 0
       T = [T 0];
   else
       T = [T 1];
   end
end

end

function plot_samples(D, T)
% D: samples 1000 x 2
% T: target 1000 x 1
% N: number of samples 1000

% blue for class 0
% red for class 1

d1 = D(1,:); d2 = D(2,:);
scatter (d1(T == 0), d2(T == 0), 'b');
hold on
scatter (d1(T == 1), d2(T == 1), 'r');
axis = [-1 1 -1 1];
legend('target 0','target 1');
title ('Samples');
xlabel ('x_1'); ylabel ('x_2');
end