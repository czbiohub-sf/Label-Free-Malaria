
% Semantic segmentation implemented in OOP form. Based heavily on the mathworks example
% found at: https://www.mathworks.com/help/vision/examples/semantic-segmentation-using-deep-learning.html

% This class can either be used directly or serve as a parent class to
% for derived specific cases (ex: WholeBloodSegmenter).

% Paul Lebel
% czbiohub
% 2019/09/27

% Requirements:
% Matlab 2019b
% Computer Vision Toolbox
% Deep Learning Toolbox
% Deep Learning Toolbox Model for selected network

%%

classdef SemanticSegmenter < handle
    properties (SetAccess = private, GetAccess = public)
        classes
        classWeights
        classtbl
        imds
        imdsTrain
        imdsVal
        imdsTest
        imageDims
        info
        pxds
        pximdsVal
        inputDir
        outputDir
        labelDir
        labelIDs
        labelMap
        netInputSize
        netName
        augmenter
        pximds
        pxdsTrain
        pxdsVal
        pxdsTest
        lgraph
        options
        tempdir
        net
        segmentedImages
        segResults
        segMetrics
    end
    
    methods(Access = public)
        
        function self = SemanticSegmenter(netName, classes, labelIDs, imageDims)
            % netName: String defining the name of the network to use.
            % Options are:  resnet18, resnet50, mobilenetv2, xception, or inceptionresnetv2
            
            % classes: A cell array containing the names of the classes
            % being segmented
            
            % labelIDs: A cell array containing elements whose order
            % matches the entries in 'classes'. Each element is an array
            % containing RGB color values
            
            % imageDims: Two-element vector containing dimensions of the
            % images, in pixels: [height, width].
            
            if verLessThan('matlab','9.7')
                error('Matlab 2019b or later is required');
            end
            
            self.classes = classes;
            self.labelIDs = labelIDs;
            self.labelMap = containers.Map(self.classes, self.labelIDs);
            
            % Specify the number of classes.
            numClasses = numel(classes);
            
            self.imageDims = imageDims;
            self.netName = netName;
            switch netName
                case 'Unet'
                    self.lgraph = unetLayers(imageDims, numClasses, 'EncoderDepth',4);
                otherwise
                    % Create DeepLab v3+.
                    self.lgraph = deeplabv3plusLayers(imageDims, numClasses, netName);
            end
            
            self.netInputSize = self.lgraph.Layers(1).InputSize;
            
        end
        
        function setInputDir(self, inputDir)
            % inputDir: directory containing images to be segmented
            self.inputDir = inputDir;
            self.imds = imageDatastore(self.inputDir, 'ReadFcn', @self.readRightColor);
        end
        
        function setOutputDir(self, outputDir)
            % outputDir: directory to write segmentation results
            self.outputDir = outputDir;
        end
        
        function setLabelDir(self, labelDir)
            % labelDir: Directory containing training labels
            self.labelDir = labelDir;
            self.pxds = pixelLabelDatastore(self.labelDir, self.classes, self.labelIDs, 'ReadFcn', @self.readRightColor);

        end
        
        function setNet(self, net, lgraph)
            self.net = net;
            self.lgraph = lgraph;
        end
        
        function balanceClassWeights(self)
            
            self.classtbl = countEachLabel(self.pxds);
            frequency = self.classtbl.PixelCount/sum(self.classtbl.PixelCount);
            imageFreq = self.classtbl.PixelCount ./ self.classtbl.ImagePixelCount;
            self.classWeights = median(imageFreq) ./ imageFreq;
            self.classWeights(isnan(self.classWeights)) = 1;
            bar(1:numel(self.classes),frequency)
            xticks(1:numel(self.classes))
            xticklabels(self.classtbl.Name)
            xtickangle(45)
            ylabel('Frequency');
            
            pxLayer = pixelClassificationLayer('Name','labels','Classes', self.classtbl.Name,'ClassWeights',self.classWeights);
            self.lgraph = replaceLayer(self.lgraph,"classification",pxLayer);
        end
        
        function setTrainingOptions(self, options)
            
            if nargin < 2
                % Define training options.
                self.options = trainingOptions('sgdm', ...
                    'LearnRateSchedule','piecewise',...
                    'LearnRateDropPeriod',10,...
                    'LearnRateDropFactor',0.3,...
                    'Momentum',0.9, ...
                    'InitialLearnRate',1e-3, ...
                    'L2Regularization',0.005, ...
                    'ValidationData',self.pximdsVal,...
                    'MaxEpochs',100, ...
                    'MiniBatchSize',4, ...
                    'Shuffle','every-epoch', ...
                    'CheckpointPath', self.tempdir, ...
                    'VerboseFrequency',2,...
                    'Plots','training-progress',...
                    'ValidationPatience', 4, ...
                    'ValidationFrequency', 10);
            else
                self.options = options();
            end
        end
        
        function enableAugmenter(self, augmenter)
            
            if nargin < 2
                % Enable data augmentation
                augmenter = imageDataAugmenter(...
                    'RandXReflection',true,...
                    'RandXTranslation',[-10 10],...
                    'RandYTranslation',[-10 10]);
            end
            
            self.augmenter = augmenter;
            self.pximds = pixelLabelImageDatastore(self.imdsTrain, self.pxdsTrain, ...
                'DataAugmentation', self.augmenter);
        end
        
        function enablePatching(self, patchSize)
            switch self.netName
                case 'Unet'
                    self.pximds = randomPatchExtractionDatastore(...
                        self.imdsTrain,self.pxdsTrain, patchSize,...
                        'DataAugmentation', self.augmenter);
                    
                    self.pximdsVal = randomPatchExtractionDatastore(...
                        self.imdsVal,self.pxdsVal, patchSize);
                otherwise
                    self.pximds = randomPatchExtractionDatastore(...
                        self.imdsTrain,self.pxdsTrain, patchSize,...
                        'DataAugmentation', self.augmenter);
                    
                    self.pximdsVal = randomPatchExtractionDatastore(...
                        self.imdsVal,self.pxdsVal, patchSize);
            end
        end
        
        function trainNet(self)
            [self.net, self.info] = trainNetwork(self.pximds,self.lgraph, self.options);
        end
        
        function testSegSingleImage(self, imgIndex)
            % Test the network on one image
            cmap = 'gray';
            I = readimage(self.imdsTest, imgIndex);
            C = semanticseg(I, self.net);
            B = labeloverlay(I,C,'Transparency',0.4);
            imshow(B)
            
            colormap(gca,cmap)
            
            % Add colorbar to current figure.
            c = colorbar('peer', gca);
            
            % Use class names for tick marks.
            c.TickLabels = self.classes;
            numClasses = size(cmap,1);
            
            % Center tick labels.
            c.Ticks = 1/(numClasses*2):1/numClasses:1;
            
            % Remove tick mark.
            c.TickLength = 0;
            
            % Compare actual to expected
            expectedResult = readimage(self.pxdsTest,imgIndex);
            actual = uint8(C);
            expected = uint8(expectedResult);
            imshowpair(actual, expected)
            
            % Compute stats
            iou = jaccard(C,expectedResult);
            table(self.classes',iou)
        end
        
        function varargout = segmentDatastore(self, dataset, minibatchsize, outputDir)
            
            if nargin < 4
                outputDir = self.outputDir;
            end
            
            if nargin < 3
                minibatchsize = 32;
            end
            
            if nargin < 2
                dataset = 'All';
            end
            
            if isa(dataset, 'char')
                
                switch dataset
                    case 'Test'
                        imDS = self.imdsTest;
                    case 'Val'
                        imDS = self.imdsVal;
                    case 'Train'
                        imDS = self.imdsTrain;
                    case 'All'
                        imDS = self.imds;
                end
                
            elseif isa(dataset, 'matlab.io.datastore.ImageDatastore') || isa(dataset, 'augmentedImageDatastore')
                imDS = dataset;
            end
            
            % Segment dataset
            self.segResults = semanticseg(imDS, self.net, ...
                'MiniBatchSize',minibatchsize, ...
                'Verbose',false, ...
                'WriteLocation', outputDir);
            
            if strcmp(dataset, 'Test')
                self.segMetrics = evaluateSemanticSegmentation(self.segResults, self.pxdsTest);
            end
            
            if nargout
                varargout{1} = self.segResults;
            end
        end
        
        function results = segmentImages(self, images)
            % Feeds the input 'images' to the semantic segmenter. Please
            % note that 'semanticseg' interprets the dimensions of 'images'
            % in a certain way, which must be complied with. 
    
            results = semanticseg(images, self.net);
        end
        
        function partitionData(self, imds, pxds)
            
            if nargin < 3
                pxds = self.pxds;
            end
            if nargin < 2
                imds = self.imds;
            end
            
            % Partition data by randomly selecting 60% of the data for training. The
            % rest is used for testing.
            
            % Set initial random state for example reproducibility.
            rng(0);
            numFiles = numel(imds.Files);
            shuffledIndices = randperm(numFiles);
            
            % Use 60% of the images for training.
            numTrain = min(numFiles-2, max(1,round(0.60 * numFiles)));
            trainingIdx = shuffledIndices(1:numTrain);
            
            % Use 20% of the images for validation
            numVal = min(numFiles-numTrain-1, max(1,round(0.20 * numFiles)));
            valIdx = shuffledIndices(numTrain+1:numTrain+numVal);
            
            % Use the rest for testing.
            testIdx = shuffledIndices(numTrain+numVal+1:end);
            
            % Create image datastores for training and test.
            trainingImages = imds.Files(trainingIdx);
            valImages = imds.Files(valIdx);
            testImages = imds.Files(testIdx);
            
            self.imdsTrain = imageDatastore(trainingImages, 'ReadFcn', @self.readRightColor);
            self.imdsVal = imageDatastore(valImages, 'ReadFcn', @self.readRightColor);
            self.imdsTest = imageDatastore(testImages, 'ReadFcn', @self.readRightColor);
            
            % Create pixel label datastores for training and test.
            trainingLabels = pxds.Files(trainingIdx);
            valLabels = pxds.Files(valIdx);
            testLabels = pxds.Files(testIdx);
            
            self.pxdsTrain = pixelLabelDatastore(trainingLabels, self.classes, self.labelIDs);
            self.pxdsVal = pixelLabelDatastore(valLabels, self.classes, self.labelIDs);
            self.pxdsTest = pixelLabelDatastore(testLabels, self.classes, self.labelIDs);
            
            % Define validation data.
            self.pximdsVal = pixelLabelImageDatastore(self.imdsVal, self.pxdsVal);
        end
        
        function img = readRightColor(self, filename)
            imgIn = imread(filename);
            nChans = self.netInputSize(3);
            
            if size(imgIn,3) == nChans
                img = imgIn;
            elseif size(imgIn,3) == 1
                img = repmat(imgIn, [1,1,3]);
            elseif size(imgIn,3) == 3
                img = rgb2gray(imgIn);
            end
        end
        
%         function C = readRightCategorical(self, filename)
%             imgIn = imread(filename);
%             nChans = self.netInputSize(3);
%             
%             if numel(self.labelIDs{1}) == nChans
%                 C = categorical(imgIn, self.labelIDs, self.classes);
%             elseif size(imgIn,3) == 1
%             end
%         end

    end
end
