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

% Should not be regularizing the theta(1) parameter (which corresponds to θ0)
theta_exclude_zero = theta;
theta_exclude_zero(1) = 0;

% 기존 costFunction 이용
[cost, grad] = costFunction(theta, X, y);

% Compute regularizing cost function
lambdaDivByM = lambda / m
J = cost + (lambdaDivByM / 2) * sum(theta_exclude_zero.^2);
grad = grad + lambdaDivByM * theta_exclude_zero;

% =============================================================

end
