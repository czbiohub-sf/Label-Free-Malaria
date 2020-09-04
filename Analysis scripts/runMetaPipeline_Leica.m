% runMetaPipeline_Leica is the same as runMetaPipeline, but adapted for
% commercial microscope data. Differences:
% - Raw images are pre-processed instead of using the MDImage class
% - User-defined variables and pipeline options are slightly different

% This is a cell script intended to be run one cell at a time.
% The overall function of this file is to step through the deep-learning based
% malaria detection pipeline from start to finish. The first section deals
% entirely with user-defined experiment specific variables that need to be set.

% Section 2 performs many steps of the pipeline in automation, although
% any one can be run or re-run separately (but you can't skip ahead). No
% arguments for most of the MetaRBCPipeline methods uses default behavior. 

% Section 3 is for human-in-loop dataset annotation. We bootstrap the label
% generation by using the best available classifier to pre-sort images
% (this makes it a lot faster for the human to sort!). Then, a fraction of
% the low confidence images are exported to disk for human sorting. Those 
% human-sorted images can be re-imported and used as corrected labels. The 
% resulting dataset is now annotated with relatively high confidence and 
% used to train a better classifier. The process can be repeated until either:
% a) The classifier performs well enough, as validated on a 100% human
% annotated ground truth dataset. Or,
% b) The entire dataset has been human-annotated.

% This section includes one more trick: once the classifier under iteration
% gets good enough, we pass an argument to
% GoogleNetClassifier.plotExampleCMInstances() indicating we would like to
% write misclassified images to disk. This trick often catches errors in
% the human annotations themselves. It's recommended to run this step and
% re-annotate the images that the classifier got wrong to ensure the ground
% truth labels were actually right. The argument is passed as the field of
% a struct:

% exInstParams.saveMisclassifiedInstances = true;
% exInstParams.saveMontage = false;
% exInstParams.montageSize = 8;
% exInstParams.saveDir = 'C:\...\Human error correction';
% myMeta.masterClassifier.plotExampleCMInstances([],[],exInstParams);

% Section 4 is an example of how to use the various plotting features in
% GoogleNetClassifier. In this case we generate the fitures from the paper.

% Author: Paul Lebel
% czbiohub
% 2020/02/28

%% 1. Set user-defined variables
pipelineOptionsPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\Leica scope\pipelineOptions.json';
baseRawDataPath = 'G:\My Drive\UV Scope\Malaria Paper\Data\Leica scope\Raw data\';
baseOutputPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\Leica scope\';

% Index identifying healthy control dataset
healthyDataset = 'SCP-2020-01-08 Healthy RBC conditions';

% Note that outputting 100% of the population still has a cap on the number of cells
% in each category (defined in pipeline settings). Doing it this way
% outputs 100% of the cells from low-frequency categories, and the
% high-frequency category (healthy) gets just the bottom N cells written. 
rbcWriteParams.method = 'lowestConfidence';
rbcWriteParams.thresholdMethod = 'percentOfPopulation';
rbcWriteParams.threshold = 100;
rbcWriteParams.maxSortedRBCsToWrite = 5000;

% Define the coarse graining parameters for analysis. "OldCats" are the names
% of the categories to be merged and "NewCats"
cg.OldCats = {{'troph', 'schizont'}};
cg.NewCats = {'late'};
cg2.OldCats = {{'ring','troph','schizont'}};
cg2.NewCats = {'parasitized'};

% By default, MetaRBCPipeline normalizes the class weights by instance
% counts. This factor imposes an additional multiplicative weight on top of
% that normalization. This is done to optimize for prior knowledge of class
% imbalance in real samples, such that the confusion matrices in real
% samples are more balanced. Note tht these factors are much smaller than
% the true class imbalance in the data.
customWeightVector = [4,2,1,1];
%% Define the MetaPipeline object and pre-process data
myMeta = MetaRBCPipeline();
myMeta.loadPipelineOptions(pipelineOptionsPath);
myMeta.setBaseRawDataPath(baseRawDataPath);
myMeta.setBaseOutputPath(baseOutputPath);
myMeta.loadMasterSegmenter();
myMeta.loadMasterClassifier();
myMeta.preprocessRawImages(); % 
myMeta.segmentMaster();
myMeta.processInstances();
myMeta.classifyWithMaster();
myMeta.mergeDatasets();
%% 3a) Human-in-loop / Export instances. 
% Repeat 3a-d) as needed to generate a fully-annotated dataset

% One can hand-annotate partial datasets and re-train classifiers to
% pre-sort larger datasets. The idea is that it is easier to hand-sort data
% that is mostly correct already, as there are fewer corrections to be
% made. Iterating this procedure a few times can result in a
% fully-annotated dataset with less effort than hand-sorting from scratch.

% Assert all the labels from the negative control dataset to be healthy.
% This also ensure that they are not written to disk in the following step
myMeta.setNegativeControlToHealthy(healthyDataset);

% Write lowest-confidence instances for the human to correct:
myMeta.writeSortedInstances(rbcWriteParams, [],[],[], true);
%% 3b)Human-in-loop / Human ensures all cells are in the correct folders %%

% Run this step after the images from 3a) have been sorted by a human
% annotator.

% Import the updated labels. This method imports the human-corrected
% labels then copies them to all corresponding RBCs in other
% channels/slices. It also updates metadata tables with the new labels.
myMeta.updateAfterHumanAnnotation(myMeta.machineSortedDirName);
myMeta.setNegativeControlToHealthy(healthyDataset);

%% 3c) Human-in-loop / Train classifiers
% Trains classifiers on the updated labels. 
% Leaving the first three arguments emtpy uses the existing datasets inside
% the Meta object for training, and creates new GoogLeNet classifiers for
% each wavelength.

% This step can take a long time (hours to days, depending on the size of
% the dataset and computing resources available)
myMeta.trainClassifiers([],[],[],customWeightVector, true);

%% 3d) Validation and analysis

% Save the new master classifier:
newMasterClassifier = myMeta.trainedClassifer{pipeOpts.masterChannelInd};
newMCPath = fullfile(myMeta.todaysPath, 'mcUnderIteration.mat');
save(newMCPath, newMasterClassifier);

% Load and set it as the new master classifier:
myMeta.loadMasterClassifier(newMCPath);

%% 3e) Validation and analysis

% Run the trained classifiers on their validation datasets
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.classifyDatastore();

% Make the plots for figure 2 and supplements:
myMeta.masterClassifier.classifyDatastore();
myMeta.masterClassifier.plotConfidenceThresholding([],[],[],[]);
myMeta.masterClassifier.plotConfidenceThresholding([],[],[],[],cg);
myMeta.masterClassifier.plotConfidenceThresholding([],[],[],[],cg2);

% Plot confidence threshold effect for raw, and two levels of
% coarse-graining. These commands produce the confusion matrices for Figure
% 2
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceThresholding([],[],[],[]);
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceThresholding([],[],[],[],cg);
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceThresholding([],[],[],[],cg2);

myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceStatistics();
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceStatistics([],[],[],[],cg);
myMeta.trainedClassifier{myMeta.pipeOpts.masterChannelInd}.plotConfidenceStatistics([],[],[],[],cg2);

%% Set new master as master classifier
myMeta.loadMasterClassifier(newMasterOutputPath);

%% Re-classify the original datasets with the new master, then merge and train on all wavelengths
myMeta.classifyWithMaster();
myMeta.mergeDatasets();
myMeta.setNegativeControlsToHealthy(healthyDataset);
myMeta.updateAfterHumanAnnotation(sortedPath);

% myMeta.trainClassifiers();
% myMeta.testClassifiers();
% myMeta.plotConfusionMatrices();

% Run the master classifier on the separate slices
for j=1:5
    [slicePred{j}, sliceProbs{j}] = myMeta.masterClassifier.classifyDatastore(myMeta.rbcInstancesSlicesSeparate{2,j});
end

[~, slConsPred, ~] = myMeta.buildConsensus(slicePred, sliceProbs);