% Calculates log-ikelihood
function [val] = log_likelihood(x, y, w)
    sigmoid_val = sigmoid_mat(x, w);
    val = sum(y.*log(sigmoid_val) + (1 - y).*log(1 - sigmoid_val));
end