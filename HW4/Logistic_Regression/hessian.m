% Calculates Hessian of Error function
function [val] = hessian(x, w)
    sigmoid_val = sigmoid_mat(x, w) ; 
    val = transpose(x) *sigmoid_val*transpose(1-sigmoid_val)*x;
end