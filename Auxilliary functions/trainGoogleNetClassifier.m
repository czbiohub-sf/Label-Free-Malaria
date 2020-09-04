function myClassifier = trainGoogleNetClassifier(inputData, baseOutputPath, options, classes, customWeightsFactor)

% This function simply feeds basic information into a GoogleNetClassifier
% object to begin the process of initialization and training.
% Paul Lebel
% czbiohub

% Inputs:
% inputData: This input can either be a path to a directory containing labeled
% folders of training images (see inputs to GoogleNetClassifier), or an
% imageDatastore object. 

% baseOutputPath: Default path to save validation results

% Default training options, if not provided
if nargin < 3 || isempty(options)
    options.miniBatchSize = 64;
    options.maxEpochs = 5;
    options.initialLearnRate = 0.01;
    options.learnRateSchedule = 'piecewise';
    options.validationFrequency = 20;
    options.learnRateDropPeriod = 2;
    options.learnRateDropFactor = 0.5;
    options.shuffle = 'every-epoch';
    options.verbose = true;
    options.plots = 'training-progress';
    options.trimLabels = false;
end

% Default class names, if not provided
if nargin < 4 || isempty(classes)
    classes = {'healthy','ring','troph','schizont'};
end

% Custom weighting factor (bias) is set to ones if not provided. 
if nargin < 5 || isempty(customWeightsFactor)
    customWeightsFactor = ones(numel(classes),1);
end

% Instantiate the classifier object
myClassifier = GoogleNetClassifier(classes, inputData, baseOutputPath);

% Display the summary of instance counts in the dataset
tbl = myClassifier.imds.countEachLabel();
disp(tbl);

% Rather than weighted loss function, you can also just get rid of extra
% training data (probably not a good idea)
if options.trimLabels
    myClassifier.trimLabels();
end

% Partitions the dataset into training and validation
myClassifier.splitLabels();

% Enable training data augmentation 
myClassifier.enableAugmenter();

% Uses weighted loss function to normalize for class count imbalances.
myClassifier.normalizeClassWeights(customWeightsFactor);

% Transfer the options to the required format
opts = trainingOptions('sgdm', ...
    'MiniBatchSize',options.miniBatchSize, ...
    'MaxEpochs', options.maxEpochs, ...
    'InitialLearnRate', options.initialLearnRate, ...
    'LearnRateSchedule', options.learnRateSchedule,...
    'LearnRateDropPeriod', options.learnRateDropPeriod, ...
    'LearnRateDropFactor',options.learnRateDropFactor, ...
    'Shuffle',options.shuffle, ...
    'ValidationData',myClassifier.augimdsVal, ...
    'ValidationFrequency',options.validationFrequency, ...
    'Verbose',options.verbose ,...
    'Plots',options.plots);

% Configure the training options, and train the network
myClassifier.setTrainingOptions(opts);
myClassifier.trainNet();
