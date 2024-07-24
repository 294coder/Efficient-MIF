classdef maeRegressionLayer < nnet.layer.RegressionLayer
    % regression layer with mean-absolute-error loss.
    
    methods
        function layer = maeRegressionLayer(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Mean absolute error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            % Calculate MAE.
            R = size(Y,3);
            flosdy = abs(Y-T);
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sum(flosdy(:))/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the MAE loss with respect to the predictions Y

%             R = size(Y,3);
            N = size(Y,4);
            dLdY = sign(Y-T)/N;
        end
    end
end