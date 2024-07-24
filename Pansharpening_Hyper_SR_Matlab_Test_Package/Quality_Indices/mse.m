function MSE = mse(X)
MSE = sum(X(:).*X(:))/numel(X);