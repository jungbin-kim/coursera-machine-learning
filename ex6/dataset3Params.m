function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Test C, sigma
%a = [0.01, 1, 30];
a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

a_length = size(a, 2);

predictions_error = zeros(a_length ^2,1);

x1 = X(:,1);
x2 = X(:,2);

for i = 1:a_length
	for j = 1:a_length
		c = a(i);
		sigma = a(j);
		model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		predictions_error(a_length*(i-1) + j) = error;
	end
end

% min error 값과 그 index 찾아 해당 C와 sigma 구함
[min_error, min_index] = min(predictions_error);
c_index = floor(min_index/a_length) + 1;
sigma_index = mod(min_index, a_length);
C = a(c_index);
sigma = a(sigma_index);


% =========================================================================

end
