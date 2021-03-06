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

G = X * theta;

H = 1 ./ (1+exp(-G));

s = sum(theta(2:1:end).^2);

mat = y.*reallog(H) + (1-y).*reallog(1-H);
J = sum(mat)/(-1*m)+ lambda/(2*m)*s;

pred = zeros(size(theta));
pred(1) = ((H - y)' * X(:,1))';

for i=2:size(theta)
	pred(i) = ((H - y)' * X(:,i))' + (lambda*theta(i));
endfor
grad = (1/m) * pred;
% =============================================================

end
