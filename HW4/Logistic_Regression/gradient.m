% Calculates gradient of error function
function [val] = gradient(x, y, w)
    sigmoid_val = sigmoid_mat(x, w) ; 
    val = transpose(x)*(sigmoid_val-y);
end