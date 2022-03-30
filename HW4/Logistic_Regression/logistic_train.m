% Trains the logistic function
function [weights] = logistic_train(data, labels, epsilon, maxiter)
    data_dim = size(data);
    weights = zeros(data_dim(2),1);
    log_max = Inf;
    log_step = log_likelihood(data, labels, weights);
    i = 0;
    while (abs(log_max) > epsilon) && (i < maxiter)
        i = i+1;
        grad = gradient(data, labels, weights);
        Hess = hessian(data, weights); 
        % Update weights
        weights = weights - (Hess)\grad;
        
        log_new = log_likelihood(data, labels, weights);
        log_max = log_step - log_new;
        log_step = log_new;
    end
    
end