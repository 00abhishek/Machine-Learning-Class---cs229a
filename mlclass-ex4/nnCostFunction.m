function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% when y is m x 1 and there are k classes, we compute Y to be m x k
% where each row has the property sum(Y(i,:)) == 1  i.e, it's a row
% of an identity matrix of size k corresponding to the value at y(i)

K = size(unique(y),1);             % K is 10, y is 5000x1
I=eye(K);
Y=I(y,:);                          % Y is 5000x10
a1 = [ones(m, 1) X];               % a1 is (5000x401)
z2 = a1 * Theta1';                 % z2 is 5000x401 * 401x25 > 5000x25
a2 = [ones(m, 1) sigmoid(z2)];     % a2 is 5000x26
z3 = a2 * Theta2';                 % z3 is 5000x26 * 26x10 > 5000x10
a3 = sigmoid(z3);                  % a3 is 5000x10

% cost represents the errors for every example cost is in m x K
cost = -(Y .* log(a3)) -((1 - Y) .* log(1 - a3));   % cost is 5000x10

% J(theta) is the the mean of the sum of errors in cost.
% the regularization is the sum of squared parameters for all layers
% times lambda/2m
J = sum(mean(cost)) + (lambda / (2 * m)) * ...
    (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^2 )));
% J is 1x1

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

d3 = a3 - Y;    % d3 is 5000x10 - 5000x10 > 5000x10
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2); % d2 is 5000x10 * 10x25 .* 5000x25 > 5000x25

Theta2_grad = (d3' * a2)/m;
Theta1_grad = (d2' * a1)/m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad = Theta2_grad + [zeros(size(Theta2, 1), 1) (lambda / m) * Theta2(:, 2:end)];
Theta1_grad = Theta1_grad + [zeros(size(Theta1, 1), 1) (lambda / m) * Theta1(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
