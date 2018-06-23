function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Calculate J
hypo_vector = X * theta;
diff_hypo_y = hypo_vector - y;
J = (1/(2*m)) * sum((diff_hypo_y) .^ 2);

% Regularized
theta_exclude_zero = theta;
theta_exclude_zero(1) = 0;

J = J + (lambda/(2*m)) * sum(theta_exclude_zero .^ 2);


% Calculate grad
grad = (1/m) * transpose(sum(diff_hypo_y .* X)) + (lambda/m) * theta_exclude_zero;


% =========================================================================

grad = grad(:);

end
