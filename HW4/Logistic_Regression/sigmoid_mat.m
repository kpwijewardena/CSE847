% Outputs the matrix containing the sigmoid outputs
function [val] = sigmoid_mat(x, w)
    z = x*w;
    val = 1.0./(1.0 + exp(-z));
end