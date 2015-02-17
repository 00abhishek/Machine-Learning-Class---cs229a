function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
errorToBeat = mean(double(svmPredict(model, Xval) ~= yval));

% fprintf('Starting with C of %f and sigma of %f\n', C, sigma);
% fprintf('The error to beat is %f\n', errorToBeat);
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;

% changed = 1;
% 
% while (changed)
%     
%     changed = 0;
    
%     minC = C-(C*0.5);
%     maxC = C+(C*0.5);
%     minSigma = sigma-(sigma*0.5);
%     maxSigma = sigma+(sigma*0.5);
%     Csteps = 10;
%     sigmaSteps = 10;

%     Crange = [minC:(maxC-minC)/(Csteps-1):maxC];
%     sigmaRange = [minSigma:(maxSigma-minSigma)/(sigmaSteps-1):maxSigma];

    Crange = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    Csteps = length(Crange);
    sigmaRange = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    sigmaSteps = length(sigmaRange);
    
    errorMatrix = zeros(Csteps,sigmaSteps);

    for a = 1:Csteps
        for b =  1:sigmaSteps
            model = svmTrain(X, y, Crange(a), @(x1, x2) gaussianKernel(x1, x2, sigmaRange(b)));
            errorMatrix(a,b) = mean(double(svmPredict(model, Xval) ~= yval));
            if (errorMatrix(a,b) < errorToBeat)
                errorToBeat = errorMatrix(a,b);
                C = Crange(a);
                sigma = sigmaRange(b);
%                 changed = 1;
            end
        end
    end
    
%     surf(Crange,sigmaRange,errorMatrix)
%     xlabel('C'); ylabel('sigma'); zlabel('error');
% end

% fprintf('Ending with C of %f and sigma of %f\n', C, sigma);
% fprintf('The error is now %f\n', errorToBeat);
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% =========================================================================

end
