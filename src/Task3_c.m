function Task3_c
% MP2 Task 3. (c)
% run this code by simply typing Task3_c in the workspace.

% The objective is to calculate the average Cross-Entropy's (CE) gradient
% w.r.t. the weight matrices (2 sets: hidden and output layers) using the
% Stochastic Error Backprop and compare the results to the ones obtained
% using the central-difference approximation to the same gradients
% Differences for all weights should be very small.

% CE: E = − [t ln(y) + (1 − t) ln(1 − y)]
% CE's gradient: ∂E/∂y = (y - t)/y(1 - y)

% The assumption is that we have solely 1 hidden layer including 2 neurons.

% Author: Maryam Najafi
% Created date: Apr 17, 2016
% Last modified date: Apr 17, 2016

close all
clc
clear all
p = [];

% set the random seed
seed = 2;
rng(seed);

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
%% 2. create the training set
D_train = D(:, 1:N_train);
%% 3. create the validation set
D_val = D(:, N_train+1:N_train + N_val);
%% 4. create the testing set
D_test = D(:, N_val+N_train+1:N);
%% 5. generate a target vector of N scalar numbers T = 1000 x 1
t = generateTarget(D, N);
%% 6. create the training set target
t_train = t(:, 1:N_train);
%% 7. create the validation set target
t_val = t(:, N_train+1:N_train + N_val);
%% 8. create the testing set target
t_test = t(:, N_val+N_train+1:N);

%%
% # of initial weight vectors
epoch = 2000;
% number of experiments for 10 different weight initializations
iters = 10;
% # of dimension
d = 2;
% # of hidden layers
L = 1;
% the number of units in each hidden layer
H = 2;
% the learning rate
etha = 0.01;
% the Cross-Entropy (cost function)
global CE; CE = [];
% step-halving
a = 0.5;

% do the following steps 10 times for 10 different initial weight vectors
for iter= 1: iters
    seed = iter;
    rng(seed); % to emphasize the difference in weight initializations
    % initialize all network weights to small random numbers
    w = generateWeightVecs(d, H, seed);
    stop = 0;
    counter = 0;
    p = [];
    etha = 0.01;
    CE = [];
    figure();
    while ~stop
        if (counter > epoch)
            break
        end
        counter = counter + 1;
        % initialize the error matrix
        d_E_EBP{1} = zeros(size(w{1})); % 2 x 1
        d_E_EBP{2} = zeros(size(w{2})); % 3 x 1
        % reset
        Y = [];
        % for every sample
        for n = 1 : N_train
            % for every layer (We have 1 hidden layer in this task)
            for l = 1 : L
                %% 9. Forward Propagation Algorithm to calculate all activation and output signals in the network
                [alpha, y, Z_out] = FP(D_train(:, n), w, L, H);
                Y = [Y; y];
                %% 10. Error Back-Propagation
                % to calculate all node errors (delta_j)^l from output layer toward the input layer
                d_E_EBP = EBP(y, alpha, t_train(n), D_train(:, n), H, w, L, Z_out, d_E_EBP);
            end
        end
        %% 11. Update weights
        [stop, w] = updateWeights(w, Y, t_train, etha, d_E_EBP, N_train);
        %p = [p; w{2}(1)];
        if (size(CE,2) > 1)
            if CE (end) > CE (end -1)
                etha = a * etha;
            else
               disp(''); 
            end
        end
    end
    plot (CE, 'g');
    legend('Cross Entropy');
    xlabel('iter'); ylabel('CE');
    title('CE variation versus iterations');
end


end

function [stop, w] = updateWeights (w, Y, t_train, etha, d_E_EBP, N_train)
global CE;
upper_limit = 10e-4;

update{1} = - etha * d_E_EBP{1}; % update weights for the hidden layer
w{1} = w{1} + update{1};
update{2} = - etha * d_E_EBP{2}; % update weights for the output layer
w{2} = w{2} + update{2};

tmp1 = abs(d_E_EBP{1}); % Norm of the gradient
tmp2 = abs(d_E_EBP{2});
max_1 = max(max(tmp1));
max_2 = max(max(tmp2));
max_norm = max(max_1, max_2);


sum = 0;
% for n = 1: N_train
%     sum = sum + (-1 * ((t_train(n) * log(Y(n))) + ((1 - t_train(n)) * log(1 - Y(n)))));
% end
CE = [CE (norm (Y -t_train'))^2];
% CE_new = sum / N_train; % the average
% CE = [CE CE_new];

if (max_norm <= upper_limit)
    stop = 1;
else
    stop = 0;
end

end

function d_E_EBP = EBP(y_n, alpha, t_n, Z_in, H, w, L, Z_out, d_E_EBP)

% error for each network output unit (in our case is 1)
delta_L = y_n - t_n; % single output so : means 1

global a; global b;
L = L + 1; % 1 hidden layer + 1 output layer
delta = [];
% error for each network hidden unit (in our case H)
for l= L - 1:-1:1
    
    if l == (L - 1)
        w_tmp = w{2};
        w_tmp = w_tmp(1:length(w_tmp) - 1);
        z_tmp = Z_out(1:size(Z_out, 1) -1);
        delta(:, l) = (1 - z_tmp.^2) .* (w_tmp * delta_L);
        %delta(:, l) = (a * b * (1 - (a * tanh(b * alpha)).^2)) .* (w_tmp * delta_L);
    else
        w_tmp = w{1};
        w_tmp = w_tmp(1:length(w_tmp) - 1);
        delta(:, l) = (a * b * (1 - (a * tanh(b * alpha)).^2)) .* (w_tmp * delta(l + 1, :));
    end
    
end

% compute the partial derivative of E w.r.t. w for 1 given sample
for l = L:-1:1
    if (l == L)
        tmp = (delta_L * Z_out')'; % Weights' erros output layer
        d_E_EBP{2} = d_E_EBP{2} + tmp; %the accumulative error for the hidden layer
    else
        tmp = (delta(:, l) * Z_in')'; % Weights' errors hidden layer
        d_E_EBP{1} = d_E_EBP{1} + tmp; % the accumulative error for the output layer
    end
    
end

end


function [alpha, y, Z_out] = FP(sample, w_0, L, H)
% Forward Propagation Algorithm
% Based on the provided notes (Error Back-Propagation Algorithm for MLP
% training)
% propagate the input forward through the network
global a; global b;
a = 1; b = 1; % given the notes

% 1. initialize with network inputs
Z_in = sample;
alpha = [];
Z_out = [];
% 2. forward-propagation signals
for l = 1 : L % For this problem only 1 hidden layer is assumed!
    alpha(:, l) = w_0{1}' * Z_in; % w_0 is a 3 by 2 matrix if
    % we have sample input of 2 plus a bias
    % and 2 hidden units in the hidden layer
    % g(activation signal) Hyperbolic Tangent Activation function plus
    % a bias weight
    Z_out(:, l) = a * tanh(b * alpha(:, l));
    
end

Z_out = [Z_out; 1]; % add bias to the z_out
y = w_0{2}' * Z_out;
% apply the logistic function to get the final output
y = 1 / (1 + exp (- y));


end

function w = generateWeightVecs(d, num_of_units, seed)
% W: a matrix of random weight values
% for the jth neuron there is a weight matrix that has 5 elements
% (W_ji)_l; for i from 1 to 5 and for only lth layer.
% Since we have only one hidden layer (H = 1), l = 1.
rng(seed);
w{1} = [];
w{2} = [];
down = 0;
up = 1;
w{1} = (up - down) * rand((d + 1), num_of_units) + down; % + 1 is bias
w{2} = (up - down) * rand((num_of_units + 1), 1) + down; % + 1 is bias

end

function D = generateSamples(N, d)
% D represents 1000 samples and stands for the Dataset
D = 2 * rand(d,N) - 1; % samples are uniformly distributed between -1 and 1
D = [D; ones(1, N)];
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
