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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part1 %

% y의 값을 1~10까지의 output 형태로 바꿔줌.
% yk를 vectorized로 해결하기 위함.
vectorizedLabel = [eye(num_labels)(:)];
convertedY = [];
for n=transpose(y),
	convertedY = [convertedY; reshape(vectorizedLabel((n-1)*num_labels+1:n*num_labels),1,num_labels)];
end;

h1 = sigmoid([ones(m,1) X] * transpose(Theta1));
h2 = sigmoid([ones(size(h1,1), 1) h1] * transpose(Theta2));

J = sum((convertedY .* log(h2) + (1 - convertedY) .* (log(1 - h2)))(:)) / -m;

% Regularized cost function
theta1_exclude_zero = Theta1;
theta1_exclude_zero(:,[1]) = []; 
theta2_exclude_zero = Theta2;
theta2_exclude_zero(:,[1]) = []; 

regularized = (lambda / (2*m)) * (sum(theta1_exclude_zero(:) .^ 2) + sum(theta2_exclude_zero(:) .^ 2));

J = J + regularized;


% Part2 %
for t=1:m,
	% Step1
	a_1 = [1; transpose(X(t,:))]; % ex. 401x1
	z_2 = Theta1 * a_1; % ex. 25x401*401x1 = 25x1
	a_2 = [1; sigmoid(z_2)]; % ex. 26x1
	z_3 = Theta2 * a_2; % ex. 10x26*26x1 = 10x1
	a_3 = sigmoid(z_3);


	% Step2
	y_t = transpose(convertedY(t,:));
	delta_3_t = a_3  - y_t; % ex. 10x1

	
	% Step3,4 set layer2
	% ex. 26x10*10x1 and then remove 1st row
	delta_2_t = (transpose(Theta2) * delta_3_t)(2:end) .* sigmoidGradient(z_2); 
	% 위의 식은 bias row 제거와 함께 한번에 계산한 것. 아래 식은 계산 후, bias row를 제거
	% delta_2_t = (transpose(Theta2) * delta_3_t) .* [1; sigmoidGradient(z_2)];
	% delta_2_t = delta_2_t(2:end); % Taking of the bias row

	Theta1_grad = Theta1_grad + delta_2_t * transpose(a_1);
	Theta2_grad = Theta2_grad + delta_3_t * transpose(a_2);

end;

% Step5
Theta1_grad = (1/m) * Theta1_grad + (lambda/m)*[zeros(size(theta1_exclude_zero,1),1) theta1_exclude_zero];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m)*[zeros(size(theta2_exclude_zero,1),1) theta2_exclude_zero];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
