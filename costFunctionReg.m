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


j1 = y' * (log(sigmoid(theta' * X')))';
j2 = (1 - y )' * (log(1 - sigmoid(theta' * X')))';
l = (lambda/m) * 0.5 * (theta(2:end)' * theta(2:end));


J = -1/m * (j1 + j2) + l;


g1_j0 = sigmoid(X * theta)' * X(:,1);
g2_j0 = y' * X(:,1);
grad_j0 = 1/m * (g1_j0 - g2_j0)';

g1_jn = sigmoid(X * theta)' * X(:,2:end);
g2_jn = y' * X(:,2:end);
grad_jn = 1/m * (g1_jn - g2_jn)' + lambda/m * theta(2:end);

grad = [grad_j0
        grad_jn];




% =============================================================

end
