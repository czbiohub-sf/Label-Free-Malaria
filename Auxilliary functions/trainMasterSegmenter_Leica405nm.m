% Test script for running the WholeBloodSegmenter class

% imgSize = [512,688];
imgSize = [2048, 2048];
patchSize = [256,256];
labelMap = containers.Map({'Background','RBC'}, [1,2]);

mySeg = WholeBloodSegmenter(patchSize, 'resnet50');

mySeg.setLabelDir('\\Flexo\MicroscopyData\Bioengineering\Cell Picker\RawData\SCP-2019-10-24 Malaria\405 nm 40x\Training Labels E2');
mySeg.setInputDir('\\Flexo\MicroscopyData\Bioengineering\Cell Picker\RawData\SCP-2019-10-24 Malaria\405 nm 40x\Training raw');
mySeg.setOutputDir('\\Flexo\MicroscopyData\Bioengineering\Cell Picker\RawData\SCP-2019-10-24 Malaria\405 nm 40x\SegOutput');


imdsIn = imageDatastore('\\Flexo\MicroscopyData\Bioengineering\Cell Picker\RawData\SCP-2019-10-24 Malaria\405 nm 40x\Raw images 8-bit');
imdsInAug = augmentedImageDatastore(imgSize, imdsIn, 'ColorPreprocessing','gray2rgb');
outputDir = '\\Flexo\MicroscopyData\Bioengineering\Cell Picker\RawData\SCP-2019-10-24 Malaria\405 nm 40x\SegOutput';

augmenter = imageDataAugmenter(...
    'RandXReflection',true,...
    'RandYReflection',true,...
    'RandXScale', [.7, 1.5], ...
    'RandYScale', [.7, 1.5], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

mySeg.partitionData();
mySeg.enableAugmenter(augmenter);
mySeg.enablePatching(patchSize);

%     'Momentum',0.9, ...
%     'L2Regularization',0.0001, ...

opts = trainingOptions('sgdm',...
    'InitialLearnRate',1E-3, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',20,...
    'LearnRateDropFactor',0.5,...
    'ValidationData',mySeg.pximdsVal,...
    'MaxEpochs',200, ...
    'MiniBatchSize',8, ...
    'CheckpointPath', mySeg.tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 10, ...
    'ValidationFrequency', 10);

mySeg.setTrainingOptions(opts);

mySeg.trainNet();

imdsLabels = mySeg.segmentDatastore(imdsInAug,1, outputDir);

mySeg.processInstances('RBC', imdsLabels, {imdsIn}, {outputDir}, 'uint16', labelMap);
