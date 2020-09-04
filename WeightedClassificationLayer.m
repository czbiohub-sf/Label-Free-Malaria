classdef WeightedClassificationLayer < nnet.layer.ClassificationLayer
               
    properties
        % Vector of weights corresponding to the classes in the training
        % data
        ClassWeights
    end

    methods
        function layer = WeightedClassificationLayer(classWeights, name)
            % layer = weightedClassificationLayer(classWeights) creates a
            % weighted cross entropy loss layer. classWeights is a row
            % vector of weights corresponding to the classes in the order
            % that they appear in the training data.
            % 
            % layer = weightedClassificationLayer(classWeights, name)
            % additionally specifies the layer name. 

            % Set class weights, ensuring that a row vector is passed
            if iscolumn(classWeights)
                classWeights = classWeights';
            end
            layer.ClassWeights = classWeights;

            % Set layer name
            if nargin > 1
                layer.Name = name;
            else
                layer.Name = 'WeightedClassificationLayer';
            end

            % Set layer description
            layer.Description = 'Weighted cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted cross
            % entropy loss between the predictions Y and the training
            % targets T.

            N = size(Y,4);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
    
            loss = -sum(W*(T.*log(Y)))/N;
        end
    end
end