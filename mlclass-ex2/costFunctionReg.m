function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sig = sigmoid(X*theta);
J = mean( - (y .* log(sig)) - ((1-y) .* log(1 - sig)) );
J = J + lambda/(2*m) * (theta(2:end)' * theta(2:end));

%n = size(X,2);
%for j = 1:n
%    s = 0;
%    for i = 1:m
%        x = X(i,:)';
%        s = s + (sigmoid(theta'*x) - y(i))*x(j);
%    end
%    grad(j) = s/m;
%    if (j>=2)
%        grad(j) = grad(j) + (lambda/m)*theta(j);
%    end
%end

grad = X'*(sigmoid(X*theta)-y)/m;
%for j = 2:size(X,2)
%    grad(j) = grad(j) + (lambda/m)*theta(j);
%end
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);

% =============================================================

end
