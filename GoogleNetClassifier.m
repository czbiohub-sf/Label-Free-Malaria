
% This class was implemented to re-train GoogLeNet and perform classification
% Based heavily on the example found at: 
% https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html

% Paul Lebel
% czbiohub

% Requirements:
% Matlab 2018b+
% Computer Vision Toolbox
% Deep Learning Toolbox
% Deep Learning Toolbox Model for selected network

%%

classdef GoogleNetClassifier < handle
    
    properties (SetAccess = private, GetAccess = public)
        augimdsTrain
        augimdsVal
        augimds
        classes
        cm
        cc
        inputDir
        imds
        imdsTrain
        imdsVal
        lgraph
        net
        netImageSize
        outputDir
        options
        probs
        splitRatio = 0.7;
        tbl
        useAugmented
        YPred
        coarseGrain = {};
    end
    
    methods(Access = public)
        
        function self = GoogleNetClassifier(classes, inputData, outputFolder, splitRatio)
            
            % classes: A cell array containing the names of the classes
            % being classified
            
            % inputDir: Directory containing folders whose names are the
            % image labels. Each folder should contain only images that
            % contain examples of that object type.
            
            % imageDims: Two-element vector containing dimensions of the
            % images, in pixels: [height, width].
            
            if nargin < 4
                self.splitRatio = 0.7;
            else
                self.splitRatio = splitRatio;
            end
            
            if verLessThan('matlab','9.5')
                error('Matlab 2018b or later is required');
            end
            
            self.setInputData(inputData);
            
            if ~exist(outputFolder,'dir')
                mkdir(outputFolder);
            end
            
            self.setOutputFolder(outputFolder);
            
            self.classes = classes;
            self.tbl = countEachLabel(self.imds);
            self.useAugmented = false;
            
            % Instantiate network
            self.net = googlenet();
            
            % Create augmentedImageDatastore from training and test sets to resize
            % images in imds to the size required by the network.
            self.netImageSize = self.net.Layers(1).InputSize;
            self.lgraph = layerGraph(self.net);
            self.lgraph = removeLayers(self.lgraph, {'loss3-classifier','prob','output'});
            
            numClasses = numel(self.classes);
            newLayers = [
                fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
                softmaxLayer('Name','softmax')
                classificationLayer('Name','classoutput')];
            self.lgraph = addLayers(self.lgraph,newLayers);
            self.lgraph = connectLayers(self.lgraph,'pool5-drop_7x7_s1','fc');
            
            % Optional freezing of the earlier layer weights
            %         layers = lgraph.Layers;
            %         connections = lgraph.Connections;
            %         layers(1:110) = freezeWeights(layers(1:110));
            %         lgraph = createLgraphUsingConnections(layers,connections);
            
        end
        
        function setInputData(self, inputData)
            
            % inputData can be either a directory name or an imageDatastore
            % object. If the former, an imageDataStore object is created
            % using the provided directory, which must comply with
            % corresponding rules as such. 
            
            switch class(inputData)
                case 'matlab.io.datastore.ImageDatastore'
                    self.imds = inputData;
                case 'char'
                    if exist(inputData,'dir')
                        self.inputDir = inputData;
                    else
                        error('Input directory does not exist!');
                    end
                    
                    self.imds = imageDatastore(inputData, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
                otherwise
                    error('Input data must either be a directory with images or an image datastore!');
            end
            
        end
        
        function setOutputFolder(self, outputDir)
            % outputDir: directory to write segmentation results
            
            if exist(outputDir,'dir')
                self.outputDir = outputDir;
            else
                error('Directory does not exist!');
            end
            
        end
        
        function setNet(self, net, lgraph)
            % Sets the network and layer graph used for classification
            self.net = net;
            self.lgraph = lgraph;
        end
        
        function setTrainingOptions(self, options)
            
            if nargin < 2
                if self.useAugmented
                    % Define training options.
                    self.options = trainingOptions('sgdm', ...
                        'MiniBatchSize',128, ...
                        'MaxEpochs',100, ...
                        'InitialLearnRate',1e-4, ...
                        'Shuffle','every-epoch', ...
                        'ValidationData',self.augimdsVal, ...
                        'ValidationFrequency',20, ...
                        'ValidationPatience',10, ...
                        'Verbose',false ,...
                        'Plots','training-progress');
                else
                    % Define training options.
                    self.options = trainingOptions('sgdm', ...
                        'MiniBatchSize',128, ...
                        'MaxEpochs',100, ...
                        'InitialLearnRate',1e-4, ...
                        'Shuffle','every-epoch', ...
                        'ValidationData',self.imdsVal, ...
                        'ValidationFrequency',20, ...
                        'ValidationPatience',10, ...
                        'Verbose',false ,...
                        'Plots','training-progress');
                end
            else
                self.options = options();
            end
        end
        
        function trimLabels(self, minSetCount)
            
            % Instead of class-balancing (recommended), one can also trim
            % the datasets to the size of the class with the least
            % examples. 
            if nargin < 2
                minSetCount = min(self.tbl{:,2});
            end
            
            self.imds = splitEachLabel(self.imds, minSetCount, 'randomize');
        end
        
        function splitLabels(self)
            
            % Split dataset into training and validation
            [self.imdsTrain, self.imdsVal] = splitEachLabel(self.imds, self.splitRatio, 'randomize');
            
        end
        
        function varargout =  enableAugmenter(self,imDS,augmenter)
            
            % Set up data augmentation. Default includes reflection,
            % translation, rotation, and scaling.
            
            % imDS: imageDatastore object from which to create the
            % augmented imageDatastore. Note: for normal usage, do not
            % provide this argument as the intent is to allow usage of this
            % as a static method. 
            
            self.useAugmented = true;
            
            if nargin < 3 || isempty(augmenter)
                
                imageAugmenter = imageDataAugmenter( ...
                    'RandXReflection',true, ...
                    'RandXTranslation',[-10 10], ...
                    'RandYTranslation',[-10 10], ...
                    'RandXScale',  [0.8, 1.3], ...
                    'RandYScale',  [0.8, 1.3], ...
                    'FillValue', (2^16)-1, ...
                    'RandRotation', [-45, 45]);
            end
            
            if nargin < 2 || isempty(imDS)
                self.augimdsTrain = augmentedImageDatastore(self.netImageSize(1:2), self.imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', imageAugmenter);
                self.augimdsVal = augmentedImageDatastore(self.netImageSize(1:2), self.imdsVal, 'ColorPreprocessing', 'gray2rgb');
                self.augimds = augmentedImageDatastore(self.netImageSize(1:2), self.imds, 'ColorPreprocessing', 'gray2rgb');
            else
                varargout{1} =  augmentedImageDatastore(self.netImageSize(1:2), imDS, 'ColorPreprocessing', 'gray2rgb');
            end
            
        end
        
        function trainNet(self)
            
            % Wrapper for trainNetwork. If data augmentation is used, a
            % different imds needs to be fed in. 
            % This step can take many hours or even days, depending on the
            % size of the dataset and the compute resources you have access
            % to. 
            
            if self.useAugmented
                self.net = trainNetwork(self.augimdsTrain, self.lgraph, self.options);
            else
                self.net = trainNetwork(self.imdsTrain, self.lgraph, self.options);
            end
        end
        
        function normalizeClassWeights(self, customWeightFactor)
            % Uses a custom class (WeightedClassificationLayer) that
            % implements weighted cross-entropy loss to normalize the class
            % weights in the event of unbalanced instance counts.
            
            % If providing a customWeightFactor, this should be a weighted
            % vector with K elements, where K is the number of classes. The
            % vector will be multiplied by the calculated class weights
            % (determined solely by instance counts), and then normalized
            % to sum to one.
            
            classTbl = self.imds.countEachLabel();
            
            if nargin < 2
                customWeightFactor = ones(numel(classTbl.Count),1);
            end
            
            % Ensure column
            if isrow(customWeightFactor)
                customWeightFactor = customWeightFactor';
            end
            
            classWeights = customWeightFactor./classTbl.Count;
            classWeights = classWeights/sum(classWeights);
            weightedClassLayer = WeightedClassificationLayer(classWeights, 'classoutput');
            try
                self.lgraph = replaceLayer(self.lgraph, 'classoutput', weightedClassLayer);
            catch
                warning('Could not replace classification layer');
            end
        end
        
        function [YPred, probs] = classifyDatastore(self, imDS, minibatchsize)
            % For large datasets, it's best to use an imageDatastore
            % object, which has all the methods used for swapping from
            % disk.
            
            % imDS: The imageDatastore object with images to be classified
            % minibatchsize: The number of individual images which get fed
            % to the GPU at once.
            
            if nargin < 3
                minibatchsize = 128;
            end
            
            if nargin < 2 || isempty(imDS)
                imDS = self.augimdsVal;
            end
            
            if ~isa(imDS, 'augmentedImageDatastore')
                testImg = imDS.readimage(1);
                if ~all(size(testImg) == self.netImageSize(1:2))
                    imDS = self.enableAugmenter(imDS);
                end
            end
            
            % Classify dataset
            [YPred, probs] = classify(self.net, imDS, 'MiniBatchSize', minibatchsize);
            self.YPred = YPred;
            self.probs = probs;
        end
        
        function cm = computeConfusionMatrix(self, predictions, gtLabels, confidences, confidenceThreshold, coarseGrain)
            
            % Computes a confusion matrix for the provided arguments (see
            % self.plotConfidenceStatistics for input argument definition) 
            
            if nargin < 6
                coarseGrain = {};
            end
            
            if nargin < 5 || isempty(confidenceThreshold)
                confidenceThreshold = 0;
            end
            
            if nargin < 4 || isempty(confidences)
                confidences = self.probs;
            end
            
            if nargin < 3 || isempty(gtLabels)
                gtLabels = self.imdsVal.Labels;
            end
            
            if nargin < 2 || isempty(predictions)
                predictions = self.YPred;
            end
            
            % Coarse grain the categories accordingly
            if ~isempty(coarseGrain)
                [tempCats, tempNums] = self.mergeCategories(coarseGrain, {gtLabels, predictions}, confidences);
                gtLabels = tempCats{1};
                predictions = tempCats{2};
                confidences = tempNums{1};
            end
                 
            keepInds = find(max(confidences,[],2) > confidenceThreshold);
            cm = confusionmat(gtLabels(keepInds), predictions(keepInds), 'Order', self.classes);
            
        end
          
        function [YPred, probs] = classifyImages(self, images)
            % Feeds the input 'images' to the classifier. Please
            % note that 'classify' interprets the dimensions of 'images'
            % in a certain way, which must be complied with.
            
            [YPred, probs] = classify(self.net, images);
        end
        
        function mtg = plotExampleCMInstances(self, imds, predictions, params)
            
            % Input arguments:
            % montageSize is a scalar value indicating the side length of
            % the montage of class examples to plot.
            
            % imds: the imageDatastore object containing the images with
            % corresponding predicted labels and ground truth labels. The
            % labels in the datastore should be the ground truth, and YPred
            % are the predicted labels.
            
            % predictions is the predicted labels for imds (cell array of strings or categorical array).
            
            fieldNames = {'coarseGrain','saveMontages','saveMisclassifiedInstances', 'montageSize', 'saveDir'};
            defaultVals = { {}, false, false, 5, self.outputDir};
            plotParams = containers.Map(fieldNames, defaultVals);
            
            % Replace default values with all or any of the supplied values
            if  nargin > 3
                if (~isempty(params))
                    for i=1:numel(fieldNames)
                        if isfield(params, fieldNames{i})
                            if ~isempty(params.(fieldNames{i}))
                                plotParams(fieldNames{i}) = params.(fieldNames{i});
                            end
                        end
                    end
                end
            end
            
            if nargin < 3 || isempty(predictions)
                predictions = self.YPred;
            end
            
            if nargin < 2 || isempty(imds)
                imds = self.imdsVal;
            end
            
            gtLabels = imds.Labels;
            
            % Coarse-grain the categories
            if ~isempty(plotParams('coarseGrain'))
                [tempCats, ~] = self.mergeCategories(plotParams('coarseGrain'), {predictions,gtLabels});
                predictions = tempCats{1};
                gtLabels = tempCats{2};
                classNames = categories(predictions);
            else
                classNames = self.classes;
            end
            
            nClasses = numel(classNames);
            confMat = confusionmat(gtLabels, predictions);
            
            % Find and plot examples from each entry in the confusion matrix
            inds = find(confMat ~= 0);
            nFigs = numel(inds);
            figs = gobjects([nFigs,1]);
            figCount = 0;
            mtg = gobjects(nClasses,nClasses);
            
            for i=1:nClasses
                for j = 1:nClasses
                    figCount = figCount + 1;
                    figs(figCount) = figure('Position',[5,5, 800, 800]);
                    
                    predLabel = classNames(i);
                    gtLabel = classNames(j);
                    
                    % Find all the corresponding entries in the
                    % confusion matrix
                    idx = find((predictions == predLabel)&(gtLabels == gtLabel));
                    
                    % Sample a random subset of the instances
                    idx = idx(randperm(numel(idx),min(plotParams('montageSize')^2, numel(idx))));
                    
                    mtg(i,j) = montage(imds,'BorderSize',3,'Indices',idx, ...
                        'Size',[plotParams('montageSize'), plotParams('montageSize')], 'BackgroundColor',...
                        [1 1 1]);
                    
                    % Save mis-classified instances. Don't bother saving
                    % them to all the possible directories. It's simpler
                    % for the user to just sort them from the gtLabel
                    % folder (this might be wrong - the point is to correct
                    % human errors, which I've found are likely to get
                    % caught by the classifier itself) into the correct
                    % gtLabel folder
                    if plotParams('saveMisclassifiedInstances') && (i~=j)
                       dirName = fullfile(plotParams('saveDir'), classNames{j});
                        if ~exist(dirName,'dir')
                            mkdir(dirName);
                        end
                        
                        fullFileNames = imds.Files(idx);
                        [~,fNames,exts] = cellfun(@fileparts, fullFileNames, 'UniformOutput',false);
                        
                        for k = 1:numel(fullFileNames)
                            copyfile(fullFileNames{k}, fullfile(dirName, [fNames{k}, exts{1}]));
                        end
                        
                    end
                    
                    axTemp = mtg(i,j).Parent;
                    
                    title(axTemp, {'Predicted: ' + string(predLabel),  "Human label: " + string(gtLabel)}, 'fontsize',22);
                    drawnow();
                    if plotParams('saveMontages')
                        try
                            saveas(figs(figCount), fullfile(params.saveDir, ['cmFig_' num2str(figCount,'%03i'),'.png']));
                        catch ME
                            disp('Could not save figure');
                            disp(ME);
                        end
                    end
                end
            end
        end
        
        function plotConfidenceStatistics(self, predictions, gtLabels, confidences, coarseGrain, saveToDisk)
            
            % Creates a tiled array of histograms of classifier confidence
            % values, arranged to correspond to a confusion matrix of the
            % same dataset. 
            
            % predictions: Categorical array of classifier predictions
            
            % gtLabels: categorical array of ground truth labels
            
            % confidences: Confidence scores for the predictions. This must
            % be an array of size (numel(predictions), numel(classNames))
            
            % coarseGrain: struct with two fields. Field 1: 'OldCats' is a
            % cell array of categories to be merge. Field 2: 'NewCats' is the
            % name of the new, merged category, or cell array of new
            % categories resulting from the merges. If 'NewCats' has
            % multiple entries, then OldCats must be a cell array of cell
            % arrays (one for each entry in NewCats). Each sub- cell array 
            % should contain old cats to be merged and re-named by the
            % corresponding entry in 'NewCats'.
            
            % saveToDisk: boolean value. 
            
            if nargin < 6 || isempty(saveToDisk)
                saveToDisk = false;
            end
            
            if nargin < 5 || isempty(coarseGrain)
                coarseGrain = {};
            end
            
            if nargin < 4 || isempty(confidences)
                confidences = self.probs;
            end
            
            if nargin < 3 || isempty(gtLabels)
                gtLabels = self.imdsVal.Labels;
            end
            
            if nargin < 2 || isempty(predictions)
                predictions = self.YPred;
            end
            
             % Coarse-grain the categories
            if ~isempty(coarseGrain)
                [tempCats, tempNums] = self.mergeCategories(coarseGrain, {predictions,gtLabels}, confidences);
                predictions = tempCats{1};
                gtLabels = tempCats{2};
                confidences = tempNums{1};
                classNames = categories(predictions);
            else
                classNames = self.classes;
            end
            
            nClasses = numel(classNames);
            
            % Create figure for overall distributions
            fig1 = figure('Position', [100,-100, 400,700]);
            for i=1:nClasses
                subplot(nClasses,1,i);
                a = zeros(nClasses, 50);
                for j=1:nClasses
                    tempProbs = confidences( (gtLabels == classNames(i)) & (predictions == classNames(j)), :);
                    probsSorted = sort(tempProbs, 2, 'descend');
                    [a(j,:), xout] = hist(probsSorted(:,1),linspace(0.5,1,50));
                end
                bar(100*xout, a);
                xlim([50,100]);
                ylim([.8, 10000]);
                ylabel('Counts','fontsize',12);
                title(classNames{i});
                set(gca, 'fontsize',12, 'YScale','log');
                legend(classNames,'Location','NorthWest');
            end
            % Label just the bottom plot's x-axis
            xlabel('Confidence (%)','fontsize',12);

            
            % Find and plot examples from each entry in the confusion matrix
            fig2 = figure('Position',[600 -100 700 700]);
            count = 0;
            
            panelSize = .8/(4+1);
            ax = gobjects(nClasses^2,1);
            
            for j=1:nClasses
                for i = 1:nClasses
                    count = count+1;                    
                    ax(count) = subplot(nClasses,nClasses,count);
                    tempProbs = confidences((predictions == classNames(j))&(gtLabels == classNames(i)), :);
                    tempProbsSorted = sort(tempProbs, 2,'descend');
                    
                    [a,xout] = hist(tempProbsSorted(:,1),100);
                    
                    bar(100*xout, a,'linewidth',2);
                    nCounts = size(tempProbs,1);
                    
                    % Set xlabels only along the bottom row
                    if j == nClasses
                        xlabel(classNames{i});
                        set(ax(count), 'XTick', [50,100]);
                        set(ax(count), 'XTickLabel',{'   50','100  '});
                    else
                        set(ax(count), 'XTick', []);
                    end
                    
                    % Set ylabels only along the left column
                    if i == 1
                        ylabel(classNames{j});
                        set(ax(count), 'YTick', [1,10,100,1000]);
                        set(ax(count), 'YTickLabel', {'1','10','100','1000'});
                    else
                        set(ax(count), 'YTick', []);
                    end
                    
                    if i == j
                        set(ax(count), 'color', [182,189,224]/255);
                    else
                        set(ax(count), 'color', [255, 219, 139]/255);
                    end
                    
                    xlim([50,100]);
                    ylim([.8, 10000]);
                    set(ax(count), 'fontsize',12, 'YScale','log');
                    
                    legStr = {['N = ', num2str(nCounts)]; ...
                        'Median = '; ...
                        [num2str(median(100*tempProbsSorted(:,1)),'%3.2f'), '%']};
                    
                    text(ax(count), 52, 1000, legStr, 'fontsize', 11);
                    drawnow();
                end
            end
            
            count = 0;
            for i=1:nClasses
                for j=1:nClasses
                    count = count+1;
                    set(ax(count), 'Position', [(j-1)*panelSize*1.15+.15, 1-(i-1)*panelSize*1.15-.25, panelSize, panelSize]);
                end
            end
            
            if saveToDisk
                try
                    saveas(fig1, fullfile(self.outputDir, 'confStatistics_overall.png'));
                    saveas(fig2, fullfile(self.outputDir, 'confStatistics_cm.png'));
                catch
                    disp('Could not save figure');
                end
            end
        end
        
        function [bestGTLabels, bestPredictions] =  plotConfidenceThresholding(self, ...
                predictions, gtLabels, confidences, coarseGrain, optimalThresMethod, popThres)
             
            % Plots the effects of thresholding the confidence values of
            % the classified predictions, including the effect of
            % coarse-graining the categories (optional).
            
            % predictions: Categorical array of classifier predictions
            
            % gtLabels: categorical array of ground truth labels
            
            % confidences: Confidence scores for the predictions. This must
            % be an array of size (numel(predictions), numel(classNames))
            
            % coarseGrain: struct with two fields. Field 1: 'OldCats' is a
            % cell array of categories to be merge. Field 2: 'NewCats' is the
            % name of the new, merged category, or cell array of new
            % categories resulting from the merges. If 'NewCats' has
            % multiple entries, then OldCats must be a cell array of cell
            % arrays (one for each entry in NewCats). Each sub- cell array 
            % should contain old cats to be merged and re-named by the
            % corresponding entry in 'NewCats'.

            % optimalThresMethod: Indicates how the threshold value will be
            % chosen when plotting the new confidence-thresholded confusion
            % matrix. Possible values are: 'lowestError' and
            % 'percentPopulation'.
            
            % popThres: If 'percentPopulation' is chosen above, then the
            % percent of the population to reject is specified here as a
            % value between (0-100). 
            
            if nargin < 7 || isempty(popThres)
                popThres = 95;
            end
            
            if nargin < 6 || isempty(optimalThresMethod)
                optimalThresMethod = 'lowestError';
            end
            
            if nargin < 5
                coarseGrain = {};
            end
            
            if nargin < 4 || isempty(confidences)
                confidences = self.probs;
            end
            
            if nargin < 3 || isempty(gtLabels)
                gtLabels = self.imdsVal.Labels;
            end
            if nargin < 2 || isempty(predictions)
                predictions = self.YPred;
            end
            
            cRange = 0:.01:1;
            ncRange = numel(cRange);

            % Coarse grain the categories accordingly
            if ~isempty(coarseGrain)
                [tempCats, tempNums] = self.mergeCategories(coarseGrain, {gtLabels, predictions}, confidences);
                gtLabels = tempCats{1};
                predictions = tempCats{2};
                confidences = tempNums{1};
            end
            
            cats = categories(gtLabels);
            gtFractions = zeros(numel(cats),1);
            for i=1:numel(cats)
                gtFractions(i) = sum(gtLabels == cats{i});
            end
            
            gtFractions = gtFractions/sum(gtFractions);
            if iscolumn(gtFractions)
                gtFractions = gtFractions';
            end
 
            classNames = categories(predictions);
            
            thresFractions = zeros(ncRange, numel(classNames));
            keptInds = cell(ncRange,1);
            metric = zeros(ncRange,1);
            nInstances = metric;
            
            for i=1:ncRange
                keptInds{i} = max(confidences,[], 2) > cRange(i);
                thresFractions(i,:) = countcats(predictions(keptInds{i}));
                thresFractions(i,:) = thresFractions(i,:)/sum(thresFractions(i,:));
                metric(i) = sqrt(mean((gtFractions - thresFractions(i,:)).^2));
                nInstances(i) = sum(keptInds{i});
            end
            
            switch optimalThresMethod
                case 'lowestError'
                    % Find the best confidence threshold
                    bestInd = find(metric == min(metric),1);
                    
                case 'percentPopulation'
                    bestInd = find((100*nInstances/max(nInstances)) < popThres/100, 1);
            end
            
            bestPredictions = predictions(keptInds{bestInd});
            bestGTLabels = gtLabels(keptInds{bestInd});            
            disp([newline, 'Ground truth fractions:']);
            for i = 1:numel(classNames)
                disp([classNames{i}, ': ', num2str(100*gtFractions(i), '%3.2f'), '%']);
            end
            
            disp([newline 'Classifier (raw) percent error: ']);
            for i=1:numel(classNames)
                disp([classNames{i}, ': ', num2str(100*(thresFractions(1,i)-gtFractions(i)), '%3.2f'), '%']);
            end
           
            disp([newline, 'Classifier (thresh) percent error:']);
            for i=1:numel(classNames)
                disp([classNames{i}, ': ', num2str(100*(thresFractions(bestInd,i)-gtFractions(i)), '%3.2f'), '%']);
            end
 
            errorVecs = 100*(thresFractions - gtFractions);
            figure('Position',[100 100 500 600]);
            subplot(2,1,1);
            plot(cRange*100, errorVecs); hold all;
            plot(cRange*100, 100*metric,'k','linewidth',2);
            plot(100*cRange(bestInd)*ones(2,1), [min(errorVecs(:)), max(errorVecs(:))], 'k--');
            legStr = classNames;
            legStr{end+1} = 'Overall';
            legend(legStr,'location','northwest');
            ylabel('Class estimation error (%)','fontsize',14);
            title('Confidence thresholding results', 'fontsize',14);
            set(gca, 'XTick', []);
            set(gca,'fontsize',12);
            axis tight;
            grid;
            
            subplot(2,1,2);
            instancesRejected = size(confidences,1) - nInstances;
            plot(cRange*100, nInstances,'linewidth',2); hold all;
            plot(cRange*100, instancesRejected,'linewidth',2);
            plot(100*cRange(bestInd)*ones(2,1), [0, max(nInstances)], 'k--');
            xlabel('Confidence threshold (%)','fontsize',14);
            ylabel('Number of instances','fontsize',14);
            legend('Instances kept','Instances rejected','location','northwest');
            ylim([-100, 1.1*max(instancesRejected)]);
            set(gca,'fontsize',12);
            grid;
            
            
            % Plot the confusion matrix for no thresholding
            figure;
            self.plotConfusionMatrix(gtLabels,predictions,'Confidence threshold = 0');
            
            % Plot the confusion matrix for optimal thresholding
            figure;
            percentRejected = 100*instancesRejected(bestInd)/max(nInstances);
            titleText = {['Confidence threshold = ', num2str(cRange(bestInd))], ...
                ['Percent rejected = ', num2str(percentRejected,'%3.1f')]};
                
            self.plotConfusionMatrix(gtLabels(keptInds{bestInd}),...
                bestPredictions, titleText);            
        end
        
        function plotConfusionMatrix(self, gtLabels, predictions, titleStr)
            % Wrapper for matlab's plotconfusion method, feeding default
            % or custom args to it. 
            if nargin < 4 || isempty(titleStr)
                titleStr = '';
            end
            
            if nargin < 3 || isempty(predictions)
                predictions = self.YPred;
            end
            
            if nargin < 2 || isempty(gtLabels)
                gtLabels = self.imdsVal.Labels;
            end
            
            plotconfusion(gtLabels, predictions, titleStr);
           
        end
        
        function statsPercent = computeStats(self, predictions, gtLabels, coarseGrain)
           % Computes classification statistics for the given predictions
           % and ground truth labels.
           
           % Inputs: 
           % predictions: categorical array of classifier predictions
           % gtLabels: categorical array of ground truth labels
           
           % Outputs:
           % statsPercent: A struct containing fields describing various
           % statistical parameters of the classification performance, all
           % in units of percent (%).
           % False positive rate (FPR)
           % Recall
           % Precision
           % Sample composition estimation error
           
           if nargin < 4
                coarseGrain = {};
           end
           
           if nargin < 3
               gtLabels = self.imdsVal.Labels;
           end
           
           if nargin < 2
               predictions = self.YPred;
           end
           
           if isrow(predictions)
               predictions = predictions';
           end
           if isrow(gtLabels)
               gtLabels = gtLabels';
           end
           
           cats = categories(gtLabels);
           
           % Convert to first capital so they concatenate in a more
           % readable manner
           for i=1:numel(cats)
               cats{i} = [upper(cats{i}(1)), lower(cats{i}(2:end))];
           end
           
           gtFractions = countcats(gtLabels);
           predFractions = countcats(predictions);
           
           % Coarse grain the categories accordingly
           if ~isempty(coarseGrain)
               [tempCats, ~] = self.mergeCategories(coarseGrain, {gtLabels, predictions});
               gtLabels = tempCats{1};
               predictions = tempCats{2};
           end
           
           CM = confusionmat(gtLabels, predictions)';
                      
           for i=1:size(CM,1)
               statsPercent.(['Precision',cats{i}]) = 100*CM(i,i)/sum(CM(i,:));
               statsPercent.(['CompErr',cats{i}]) = 100*(predFractions(i) - gtFractions(i))/sum(gtFractions);
           end
           
           for j = 1:size(CM,2)
               statsPercent.(['Recall', cats{j}]) = 100*CM(j,j)/sum(CM(:,j));
               for i=1:size(CM,1)
                   if i~=j
                       statsPercent.(['FPR',cats{j},cats{i}]) = 100*CM(i,j)/sum(CM(:,j));
                   end
               end
           end
           
           statsPercent.ovAcc = 100*trace(CM)./sum(sum(CM));
            
        end
        
        function [categoricals, numericals] = mergeCategories(~, coarseGrain, categoricals, numericals)
            % Merges categories in categorical arrays according to the struct
            % coarseGrain: struct with two fields. Field 1: 'OldCats' is a
            % cell array of categories to be merge. Field 2: 'NewCats' is the
            % name of the new, merged category, or cell array of new
            % categories resulting from the merges. If 'NewCats' has
            % multiple entries, then OldCats must be a cell array of cell
            % arrays (one for each entry in NewCats). Each sub- cell array 
            % should contain old cats to be merged and re-named by the
            % corresponding entry in 'NewCats'.
            
            % categoricals: categorical array or cell array of categoricals
            % to be merged.
            
            % numericals: numeric array or cell array of numerical arrays
            % to be merged. Note that it is assumed that
            % size(numericals{i},2) == numel(categories(categoricals{i}))
            % and that each column of every numericals{i} corresponds to
            % the corresponding category of every categoricals{i}.
                        
            if nargin < 4 || isempty(numericals)
                numericals = [];
            end
            
            if nargin < 3 || isempty(categoricals)
                categoricals = categorical();
            end
            
            % Convert all to cell for indexing
            if ~iscell(coarseGrain.NewCats)
                coarseGrain.NewCats = {coarseGrain.NewCats};
            end
            if ~iscell(categoricals)
                categoricals = {categoricals};
            end
            if ~iscell(numericals)
                numericals = {numericals};
            end
            
            classNames = categories(categoricals{1});
            
            for i = 1:numel(coarseGrain.NewCats)
                
                for j = 1:numel(categoricals)
                    if ~isempty(categoricals{j})
                        categoricals{j} = mergecats(categoricals{j}, coarseGrain.OldCats{i}, coarseGrain.NewCats{i});
                    end
                end
                
                % Find the column indices corresponding to the OldCats
                catInds = contains(classNames, coarseGrain.OldCats{i});
                
                
                for k=1:numel(numericals)
                    if ~isempty(numericals{k})
                        % Average over the inds for the categories to be
                        % merged, and store the result in the first such
                        % occuring index. Delete the columns corresponding to
                        % the other old categories
                        firstInd = find(catInds,1);
                        numericals{k}(:,firstInd) = sum(numericals{k}(:,catInds),2);
                        otherInds = catInds;
                        otherInds(firstInd) = 0;
                        numericals{k}(:,otherInds) = [];
                    end
                end
                
                % Update classNames as the number gets reduced!
                classNames = categories(categoricals{1});
            end
        end
        
        function [gtFractions, predictedFractions, predictedFractionsWeighted] = compareOverallFractions(self, predictions, confidences, gtLabels, coarseGrain)
           
            % Compares the overall prediction of the dataset's composition
            % for each category, with that of the ground truth labels. The
            % prediction is computed in two ways: first, the simple
            % summation of 'winner takes all' classifier output (each
            % instance scored as the category with the highest confidence
            % score). Second, the overall composition is also computed very
            % simply by summing the confidence score across all instances.
            % This method allows for each instance to have a weighted
            % contribution to the overall estimate, depending on the
            % classifier's confidence score.
            
            if nargin < 5 || isempty(coarseGrain)
                coarseGrain = {};
            end
            
            if nargin < 4 || isempty(gtLabels)
                gtLabels = self.imdsVal.Labels;
            end
            
            if nargin < 3 || isempty(confidences)
                confidences = self.probs;
            end
            
            if nargin < 2 || isempty(predictions)
                predictions = self.YPred;
            end
            
             % Coarse grain the categories accordingly
            if ~isempty(coarseGrain)
                [tempCats, tempNums] = self.mergeCategories(coarseGrain, {gtLabels, predictions}, confidences);
                gtLabels = tempCats{1};
                predictions = tempCats{2};
                confidences = tempNums{1};
            end
            
            gtFractions = countcats(gtLabels)/sum(countcats(gtLabels));
            
            % Method 1
            predictedFractions = countcats(predictions)/sum(countcats(predictions));
            
            % Method 2
            predictedFractionsWeighted = mean(confidences,1)';
            
        end

        
    end
    
end

