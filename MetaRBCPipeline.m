% Purpose:
% This class uses Matlab's Deep Learning and Computer Vision toolboxes to
% train classifiers to recognize objects (red blood cells) using various
% wavelengths of light and/or focus slices of a z-stack, providing the
% infrastructure to track performance across these variables. 

% Although multiple color channels and/or focus slices are not required,
% a master wavelength and best focus slice can serve as master dataset in
% order to generate training labels for the other wavelengths/slices.
% The raw data from multiple datasets can be processed together or
% separately in order to access more training data.

% Sequence of steps performed:

% 1. Instantiate object, provide file paths to raw data, pipeline options
%   (.json file), load segmenter, load master classifier.

% 2. Pre-processing:
    % Find best focus slices.
    % Do affine or cross-corr transformation to align colors/slices.
    % Rescale dynamic range.
    % Export preprocessed images.
    % Create Matlab imageDatastores for all these new files.

% 3. Do semantic segmentation on the master color/slice, save the label
%    matrices.

% 4. Process all the RBC instances resulting from applying the label
%       matrices to all the original datasets (colors, slices).

% 5. Create imageDatastores from all these instances (for each color, slice,
%       dataset).

% 6. Classify these instances for the master color and slice, using the
%       master classifier.

% 7. Apply the resulting labels to all the corresponding instances in the
%       other slices/channels.

% 8. Merge all datasets (keeping wavelengths separate), forming new
%       imageDatastores.

% 9. Export a low-confidence subset of the machine-sorted images into separate
%     directories for humam sorting. Human corrects any errors by dragging
%     images into the correct folders.

% 10. Import the results of the human sorting on the master dataset and
%     propagate the labels to all the other channels/slices. Also update
%     the metadata tables. Maintains a new dataStore composed of 100%
%     human-annotated data.

% 11. Write updated metadata files (including bounding boxes to provide
%     training coordinates to external algorithms.

% 12. Train new GoogLeNet classifiers (one for each wavelength) on all
%       available data (merged).

% 13. Validate classifier performance as a function of wavelength and slice by
%       classifying their respective validation datasets and plotting
%       confusion matrices, and other metrics.

% Repeat steps 9-13 as necessary


% 14. Evaluate further by plotting confidence statistics, thresholding,
% consensus scoring, etc...

% Author: Paul Lebel
% czbiohub

% Using the following custom classes and functions:
% GoogleNetClassifier, SemanticSegmenter, MDImage, WholeBloodSegmenter, combineIMDS

% Requirements:
% Matlab 2018b+
% Computer Vision Toolbox
% Deep Learning Toolbox
% Deep Learning Toolbox Model for selected network (Only GoogLeNet has been implemented)


classdef MetaRBCPipeline < handle
    
    properties (SetAccess = private, GetAccess = public)
        
        % Base file paths and common parameters
        pipelineOptionsPath
        baseRawDataPath
        baseOutputPath
        rawDataIDs
        rawDataPaths
        rawDataImds
        todaysPath
        nDatasets
        pipeOpts
        nChannels
        nSlices
        nTimes
        nPositions
        classes
        nClasses
        
        % GoogleNetClassifier object instance
        masterClassifier
        
        % WholeBloodSegmenter object instance
        rbcSegmenter
        
        % MDImage object instance
        mdImage
        
        % Pre-processed raw image files and datastores
        preprocFilenamesCleaned
        preprocImds
        preprocImdsAug
        preprocOutputDir
        validFocusInds
        
        % RBC instance directories, datastores, augmented image datastores, and metadata
        rbcLabelOutputDir
        rbcInstanceDir
        rbcInstanceImds
        rbcInstanceAug
        instanceMetadataRaw
        instanceMetadata
        instanceOptions
        sortedMasterInstanceDir
        xlsFullFileName
        
        % Cell array of image datastores for the master label matrices
        % (raw segmentation output)
        masterLabelMatrix
        
        % Classifier output related
        masterClassPrediction
        probs
        mergedProbs
        classifierOutputDir
        
        % Merged datastores and metadata
        rbcInstancesMerged
        rimLabelType
        rbcInstancesSlicesSeparate
        rissLabelType
        mergedMetadataSlicesSeparate
        rbcInstancesSlicesSeparateAug
        mergedMetadata
        rbcInstancesMergedAug
        machineSortedDirName
        
        % Trained classifier related
        trainedClassifier
        trainedYPred
        trainedYPredMerged
        trainedProbs
        trainedProbsMerged
        
        % Figure handles and results statistics
        resultsFig
        cm
        trainedCMSlicesSeparate
        trainedCMMerged
        
        % Human-in-loop
        humanOnlyMetadata
        humanSortedImds
        rbcInstancesHumanOnly
        sortedExportType
        
    end
    
    properties (Constant, Access = private)
        % Define output subfolder names
        RBCLabelsFolderName = 'Matlab-RBC Label Matrix';
        RBCInstancesOutputFolderName = 'Matlab-RBC Instances';
        sortedRBCFolderName = 'Machine-Sorted RBCs';
        preprocFolderName = 'RefocusedRegistered';
        mergedResultsFolderName = 'Merged Training Results';
        
        % Required steps in the pipeline
        stepList = {'Load pipeline options', 'Set base raw data path',...
            'Set base output path','Load master segmenter','Load master classifier',...
            'Preprocess Images','Segment master','Process Instances'};
        
        maxSpreadsheetRows = 25000;
        
        dataDependentFields = {...
            'preprocFilenamesCleaned',...
            'preprocImds',...
            'preprocImdsAug',...
            'validFocusInds',...
            'instanceMetadataRaw',...
            'rbcInstanceDir',...
            'rbcInstanceImds',...
            'rbcInstanceAug',...
            'nChannels',...
            'pipeOpts',...
            'xlsFullFileName',...
            'rbcInstancesMerged',...
            'rbcInstancesSlicesSeparate',...
            'mergedMetadataSlicesSeparate',...
            'rbcInstancesSlicesSeparateAug',...
            'mergedMetadata',...
            'rbcInstancesMergedAug',...
            'trainedClassifier',...
            'trainedYPred',...
            'trainedProbs',...
            'trainedProbsMerged',...
            'trainedYPredMerged',...
            'trainedCMSlicesSeparate',...
            'trainedCMMerged'};
    end
    
    methods(Access = public)
        
        function self = MetaRBCPipeline()
            % Constructor doesn't need to initialize anything.
        end
        
        function loadPipelineOptions(self, pipelineOptionsPath)
            
            % A .json file containing experiment-specific settings is used
            % for configuration
            
            self.pipelineOptionsPath = pipelineOptionsPath;
            
            % Read the json file and decode it
            fid = fopen(self.pipelineOptionsPath,'r');
            rawData = fread(fid, '*char');
            ops = jsondecode(rawData');
            fclose(fid);
            self.pipeOpts = ops.pipelineOptions;
            
            % Make this field more accessible by moving it up and removing
            % existing one to avoid confusion
            self.instanceOptions = self.pipeOpts.instanceOptions;
            self.pipeOpts = rmfield(self.pipeOpts, 'instanceOptions');
            
            % Convert labelMap to a map
            keys = fields(self.instanceOptions.labelMap);
            tempVals = cell(numel(keys),1);
            for i=1:numel(keys)
                tempVals{i} = self.instanceOptions.labelMap.(keys{i});
            end
            self.instanceOptions.labelMap = containers.Map(keys, tempVals);
            
            self.nChannels = self.pipeOpts.nChannels;
            self.nSlices = self.pipeOpts.nSlices;
            
            self.sortedMasterInstanceDir = fullfile(self.todaysPath, self.sortedRBCFolderName, self.pipeOpts.wavelengthLabels{self.pipeOpts.masterChannelInd});
            if ~exist(self.sortedMasterInstanceDir,'dir')
                mkdir(self.sortedMasterInstanceDir);
            end
            
            % Save a copy of the pipeline options that were used in this
            % instance, as the original file may change.
            copyfile(self.pipelineOptionsPath, fullfile(...
                self.todaysPath, 'pipelineOptions.json'));
        end
        
        function setBaseRawDataPath(self, baseRawDataPath, chooseSubset)
            % This method allows the user to set one base raw data path
            % containing several raw dataset directories at once. 
            % The user can then choose subset (or all) of the datasets for 
            % processing. 
            
            % Inputs:
            % baseRawDataPath is a character string of the base
            % directory that contains all the individual directories; one
            % for each dataset.
            %
            % chooseSubset: bool value indicating whether the user wants to
            % choose only a subset of the datasets.
            
            
            if nargin < 3
                chooseSubset = true;
            end
            
            if exist(baseRawDataPath,'dir')
                self.baseRawDataPath = baseRawDataPath;
            else
                error('Directory does not exist!');
            end
            
            dirList = dir(baseRawDataPath);
            % Start enumerating the directory list at 3 because dir()
            % returns '.' and '..' as the first two elements.
            dirList = {dirList(3:end).name};
            
            if chooseSubset
                [indx,tf] = listdlg('PromptString', 'Select all dataset directories to include in the analysis.',...
                    'SelectionMode','multiple','ListString',dirList);
                if tf
                    dirList = dirList(indx);
                else
                    warning('User cancelled selection operation. No datasets were selected')
                    return
                end
            end
            
            % Add all the sub-folders listed in dirList
            for i = 1:numel(dirList)
                fullPath = fullfile(self.baseRawDataPath, dirList{i});
                if exist(fullPath, 'dir')
                    self.rawDataPaths{i} = fullPath;
                end
            end
            
            self.rawDataIDs = dirList;
            self.nDatasets = numel(self.rawDataIDs);
        end
        
        function setBaseOutputPath(self, baseOutputPath)
            
            % Sets the default path for all outputs including pre-processed data,
            % figures, and metadata. A path is created within the base
            % output path with a time-stamp, ensuring that each time the
            % pipeline is run, a unique directory is used. 
            
            % baseOutputPath is a string representing the path to the
            % desired output directory. 
            
            if ~exist(baseOutputPath,'dir')
                try
                    [status, msg] = mkdir(baseOutputPath);
                catch
                end
                
                if ~status
                    disp(msg);
                    error('Could not create baseOutputPath');
                end
            end
            
            self.baseOutputPath = baseOutputPath;
            self.todaysPath = fullfile([baseOutputPath, 'CrossTrain-', datestr(now,'yyyy-mm-dd-HH-MM-SS')]);
            
            try
                [status, msg] = mkdir(self.todaysPath);
            catch
            end
            if ~status
                disp(msg);
                error('Could not create baseOutputPath directory');
            end
            
        end
        
        function loadMasterSegmenter(self)
            % Load rbc segmenter
            temp = load(self.pipeOpts.masterSegmenterPath,'mySeg');
            self.rbcSegmenter = temp.mySeg;
            self.rbcSegmenter.setAreaRange('RBC',self.pipeOpts.areaRange);
            self.rbcSegmenter.setTileCanvasSize(self.pipeOpts.rbcTileSizeClassification);
        end
        
        function loadMasterClassifier(self, mPath)
            
            % mPath: Path to GoogleNetClassifier to set as master. The
            % first object of the right class will be loaded from the .mat
            % file that is specified in mPath.
            
            if nargin < 2
                mPath = self.pipeOpts.masterClassifierPath;
            end
            
            % Load masterClassifier
            temp = load(mPath);
            fNames = fields(temp);
            
            for i=1:numel(fNames)
                if isa(temp.(fNames{i}), 'GoogleNetClassifier')
                    self.masterClassifier = temp.(fNames{i});
                end
            end
            
            self.nClasses = numel(self.masterClassifier.classes);
            self.classes = self.masterClassifier.classes;
            self.pipeOpts.masterClassifierPath = mPath;
        end
        
        function preprocessMDImages(self, datasetInds)
            % preprocessMDImages assumes that each self.rawDataIDs is a
            % directory that contains a .mat file with an MDIMage object,
            % which has all the dataset's metadata self-contained, but must
            % also match the pipelineOptions.json file. The MDImage file
            % will load all the raw data, assumed to be in the same
            % directory as the file itself.
            
            % After loading MDImage, this method steps through a few of the
            % pre-processing steps:
            % 1. Load raw images
            % 2. Re-focus
            % 3. Register channels
            % 4. Export pre-processed images
            
            % Inputs: (optional) datasetInds are indices of the datasets
            % the user wishes to pre-process. Normally this would be left
            % empty to do them all, but in the case where new data is added
            % later this is useful.
            if nargin < 2
                datasetInds = 1:self.nDatasets;
            end
            
            % Loop through all the MDImages, one from each raw dataset
            for m = datasetInds
                
                disp('--------------------------------------------------');
                disp('Preprocessing images...');
                disp(['Dataset ' num2str(m) ': ' self.rawDataIDs{m}]);
                
                % Load the MDImage object. Matlab lumps all variables in a .mat file into a single
                % struct. We will loop through until we find the first
                % 'MDImage' object and then use it.
                MDFilename = ls([self.rawDataPaths{m} '\*MDImage.mat']);
                tempStruct = load(fullfile(self.rawDataPaths{m}, MDFilename));
                
                % UV Scope MDA saves the MDImage variable name as tempImage
                fNames = fields(tempStruct);
                for i=1:numel(fNames)
                    temp = tempStruct.(fNames{i});
                    if isa(temp,'MDImage')
                        self.mdImage = temp;
                        break;
                    end
                end
                
                % Extract the file prefix and re-use it as it won't change
                self.mdImage.setFilePath(self.rawDataPaths{m}, self.mdImage.filePrefix);
                
                % Read the metadata in case the dataset was aborted early
                self.mdImage.importMetadata(fullfile(self.rawDataPaths{m}, [self.mdImage.filePrefix, '_metadata.json']));
                
                % In case there are corrected images already present, unload them
                self.mdImage.unloadCorrectedImages(true);
                disp('Loading raw images...');
                
                if ~self.mdImage.imagesLoaded
                    self.mdImage.loadImages('tiff',[],[],[],[],true);
                end
                
                % Refocus the images, using parfocal correction
                disp('Refocusing raw images...');
                
                if any(contains(fields(self.pipeOpts), 'focusMethod'))
                    focusMethod = self.pipeOpts.focusMethod;
                else
                    focusMethod = 'gradient';
                end
                
                self.mdImage.refocusImages(focusMethod,floor(self.nSlices/2),'all', self.pipeOpts.parfocalCorrection);
                
                % validFocusInds{m} is an array with dimensions (channel, position,
                % time)
                self.validFocusInds{m} = self.mdImage.focusValid;
                self.mdImage.scaleDynamicRange('corrected',16);
                
                % Perform affine transformations to align channels
                disp('Performing affine transformation...');
                self.mdImage.registerChannels(self.pipeOpts.masterChannelInd);
                
                % Write the refocused, aligned images to files
                self.preprocOutputDir{m} = fullfile(self.todaysPath, self.rawDataIDs{m}, self.preprocFolderName);
                disp('Writing preprocessed images to disk...');
                tempFilenames = self.mdImage.exportCorrectedImages(self.preprocOutputDir{m}, self.mdImage.filePrefix, true, true);
                
                % Create an imds and AugImds for each slice and channel
                self.nPositions(m) = size(tempFilenames, 3);
                self.nTimes(m) = size(tempFilenames,4);
                
                for i=1:self.nChannels
                    for j = 1:self.nSlices
                        % MDImage uses [slices, channels, positions, times]
                        % but here we use [channels, slices, positions, times]. Swap the dimension order so that it matches.
                        % Then squash the positions and times together as we treat them the
                        % same here.
                        self.preprocFilenamesCleaned{m,i,j} =  reshape(tempFilenames(j,i,:,:), [self.nPositions(m)*self.nTimes(m),1]);
                        
                        % Remove empty entries, which correspond to channels/positions/times
                        % within each dataset that did not achieve proper focus
                        self.preprocFilenamesCleaned{m,i,j} = self.preprocFilenamesCleaned{m,i,j}(~cellfun('isempty', self.preprocFilenamesCleaned{m,i,j}));
                        self.preprocImds{m,i,j} = imageDatastore(squeeze(self.preprocFilenamesCleaned{m,i,j}),'ReadFcn', @convertTo8bit);
                        self.preprocImdsAug{m,i,j} = augmentedImageDatastore(self.mdImage.frameDims, self.preprocImds{m,i,j}, 'ColorPreprocessing', 'gray2rgb');
                    end
                end
            end
        end
        
        function preprocessRawImages(self,datasetInds)
            % preprocessRawImages assumes that each self.rawDataIDs is a
            % directory that contains sub-directories titled with the
            % wavelength labels specified in the pipelineOptions.json file.
            % imageDatastore objects are created for each of the
            % wavelengths specified in the .json file.
            
            % At this time I did not support pre-processing focus stacks in
            % this method but it would be very easy to implement.
            
            % Inputs: (optional) datasetInds are indices of the datasets
            % the user wishes to pre-process. Normally this would be left
            % empty to do them all, but in the case where new data is added
            % later this is useful.
            
            if nargin < 2
                datasetInds = 1:self.nDatasets;
            end
            
            % Create directories and image datastores
            for m=datasetInds
                disp('--------------------------------------------------');
                disp('Preprocessing images...');
                disp(['Dataset ' num2str(m) ': ' self.rawDataIDs{m}]);
                
                for i=1:self.nChannels
                    self.preprocOutputDir{m,i} = fullfile(self.todaysPath, self.rawDataIDs{m}, [self.pipeOpts.wavelengthLabels{i}, '_aligned']);
                    if ~exist(self.preprocOutputDir{m,i}, 'dir')
                        mkdir(self.preprocOutputDir{m,i});
                    end
                    
                    % Create an imageDatastore from each raw data path,
                    % assuming the sub-folders are named after the
                    % wavelengthLabels
                    self.rawDataImds{m,i} = imageDatastore(fullfile(self.rawDataPaths{m}, self.pipeOpts.wavelengthLabels{i}));
                end
                
                [optimizer, metric] = imregconfig('multimodal');
                
                % Align each of the slices to the master channel
                for i=1:self.nChannels
                    for p = 1:numel(self.rawDataImds{m,i}.Files)
                        disp(['Aligning dataset ' num2str(m), ', channel ' num2str(i), ', image ' num2str(p)]);
                        
                        imgFixed = convertTo16bit(self.rawDataImds{m,self.pipeOpts.masterChannelInd}.Files{p});
                        imgMoving = convertTo16bit(self.rawDataImds{m,i}.Files{p});
                        
                        switch self.pipeOpts.alignmentMode
                            case 'affine'
                                imgTformed = imregister(imgMoving,imgFixed, 'affine', optimizer, metric);
                            case 'xcorr'
                                [~,~,imgTformed] = imageCrossCorr(imgFixed, imgMoving, false, false, false, true, (2^self.pipeOpts.rawImageBitDepth -1));
                        end
                        
                        [~,fName, ext] = fileparts(self.rawDataImds{m,i}.Files{p});
                        imwrite(uint16(imgTformed), fullfile(self.preprocOutputDir{m,i}, [fName, ext]));
                    end
                    
                    % Create imageDatastores from the exported data. We
                    % convert to 8-bit upon reading the raw frames for
                    % segmentation only.
                    self.preprocImds{m,i} = imageDatastore(self.preprocOutputDir{m,i}, 'ReadFcn', @convertTo8bit);
                    
                    % Force dimension vector to be a row because
                    % augmentedImageDatastore input dims is brittle and
                    % errors with a column vector.
                    rawDims = self.pipeOpts.rawDataFrameDims;
                    if iscolumn(rawDims)
                        rawDims = rawDims';
                    end
                    
                    self.preprocImdsAug{m,i} = augmentedImageDatastore(rawDims, self.preprocImds{m,i},'ColorPreprocessing', 'gray2rgb');
                    
                end
            end
        end
        
        function segmentMaster(self, mInds)
            
            % Performs semantic segmentation on the best-focus, master 
            % wavelength images, as specified in self.pipeOpts. 
            
            % Optional input mInds is a vector of indices into the
            % rawDataIDs to be process. Default: all datasets will be
            % processed. 
            
            if nargin < 2 || isempty(mInds)
                mInds = 1:self.nDatasets;
            end
            
            disp('--------------------------------------------------');
            disp('Segmenting images...');
            
            for m = mInds
                
                disp(['Dataset ' num2str(m) ': ' self.rawDataIDs{m}]);
                
                % Segment the master channel/slice and use it for all the other channels/slices
                self.rbcLabelOutputDir{m} = fullfile(self.todaysPath, self.rawDataIDs{m}, self.RBCLabelsFolderName, ...
                    self.pipeOpts.wavelengthLabels{self.pipeOpts.masterChannelInd}, self.pipeOpts.sliceLabels{self.pipeOpts.masterSliceInd});
                
                if ~exist(self.rbcLabelOutputDir{m},'dir')
                    mkdir(self.rbcLabelOutputDir{m});
                end
                
                self.masterLabelMatrix{m} = self.rbcSegmenter.segmentDatastore(self.preprocImdsAug{m,self.pipeOpts.masterChannelInd,...
                    self.pipeOpts.masterSliceInd}, self.pipeOpts.segMiniBatchSize, self.rbcLabelOutputDir{m});
            end
        end
        
        function processInstances(self, datasetInds)
            
            % This method uses the semantic segmentation maps to produce
            % thumbnails of RBC images for multiple wavelengths and slices.
            % The thumbnails are all saved individually with unique
            % filenames and added to new datastores. 
            
            % Optional input argument specifies a specific raw dataset on
            % which to perform the operation. Otherwise, all datasets will
            % be processed. 
            
            if nargin < 2 || isempty(datasetInds)
                datasetInds = 1:self.nDatasets;
            end
            
            disp('--------------------------------------------------');
            disp('Processing instances...');
            
            % Process the RBC instances from all color channels and slices using the segmentations from the best focus channel/slice.
            for m = datasetInds
                disp(['Dataset ' num2str(m) ': ' self.rawDataIDs{m}]);
                
                disp('Creating rbc instance directories');
                for i = 1:self.nChannels
                    for j = 1:self.nSlices
                        self.rbcInstanceDir{m,i,j} = fullfile(self.todaysPath, self.rawDataIDs{m}, self.RBCInstancesOutputFolderName, self.pipeOpts.wavelengthLabels{i}, self.pipeOpts.sliceLabels{j});
                        self.classifierOutputDir{m,j} = fullfile(self.todaysPath, self.rawDataIDs{m}, self.RBCInstancesOutputFolderName, self.pipeOpts.wavelengthLabels{i});
                        if ~exist(self.rbcInstanceDir{m,i,j},'dir')
                            mkdir(self.rbcInstanceDir{m,i,j});
                        end
                    end
                end
                
                self.instanceOptions.filePrefix = self.rawDataIDs{m};
                
                disp('Performing instancing...');
                [~, self.instanceMetadataRaw{m}] = self.rbcSegmenter.processInstances(self.masterLabelMatrix{m}, ...
                    reshape(self.preprocImdsAug(m,:,:), [self.nChannels*self.nSlices,1]), ...
                    reshape(self.rbcInstanceDir(m,:,:), [self.nChannels*self.nSlices,1]), ...
                    self.instanceOptions);
                
                % Create datastores from the segmented RBC instances
                disp('Creating datastores for the instances...');
                for i = 1:self.nChannels
                    for j = 1:self.nSlices
                        self.rbcInstanceImds{m,i,j} = imageDatastore(...
                            self.rbcInstanceDir{m,i,j},'ReadFcn',@self.convertTo16bit);
                        
                        self.rbcInstanceAug{m,i,j} = augmentedImageDatastore(...
                            self.masterClassifier.netImageSize(1:2),...
                            self.rbcInstanceImds{m,i,j}, 'ColorPreprocessing', 'gray2rgb');
                    end
                end
            end
        end
        
        function classifyWithMaster(self,datasetInds)
            
            % Use the master classifier to classify all the image
            % datastores
            
            % Optional input:
            % datasetInds: a vector of indices indicating which raw
            % datasets to classify. If not provided, all of them will be
            % classified.
            
            if (nargin < 2 ) || isempty(datasetInds)
                datasetInds = 1:self.nDatasets;
            end
            
            % Perform classification on just the master channel/slice
            disp('--------------------------------------------------');
            disp('Classifying datasets...');
            
            for m = datasetInds
                disp(['Dataset ' num2str(m) ': ' self.rawDataIDs{m}]);
                [self.masterClassPrediction{m}, self.probs{m}] = self.masterClassifier.classifyDatastore(...
                    self.rbcInstanceAug{m,self.pipeOpts.masterChannelInd, self.pipeOpts.masterSliceInd},...
                    self.pipeOpts.classMiniBatchsize);
                
                % Unflatten the metadata to separately index
                % slices/channels
                self.instanceMetadataRaw{m} = reshape(self.instanceMetadataRaw{m}, [self.nChannels, self.nSlices]);
                
                % Apply master labels to all other slices/channels
                for i = 1:self.nChannels
                    for j = 1:self.nSlices
                        self.instanceMetadata{m,i,j} = self.instanceMetadataRaw{m}{i,j};
                        self.instanceMetadata{m,i,j}.masterClassPrediction = self.masterClassPrediction{m};
                        self.xlsFullFileName{m,i,j} = ...
                            fullfile(self.todaysPath, ...
                            self.rawDataIDs{m},...
                            self.RBCInstancesOutputFolderName, ...
                            self.pipeOpts.wavelengthLabels{i}, ...
                            ['instanceMetadata_', self.pipeOpts.sliceLabels{j}, '.xls']);
                    end
                end
                
                % Apply labels to all the above image datastores
                for i = 1:self.nChannels
                    for j = 1:self.nSlices
                        self.rbcInstanceImds{m,i,j}.Labels = self.masterClassPrediction{m};
                    end
                end
            end
        end
        
        function mergeDatasets(self)
            
            % Creates merged imageDatastore objects comprising images from
            % all the raw datasets in the pipeline, including pooled focus
            % slices (but not wavelengths), as well as merged but keeping
            % slices separate. This operation should always be perfomred
            % after ingesting and pre-processing raw data. 
            
            disp('Merging datasets...');
            
            % Create merged imds' that include all the datasets
            for i = 1:self.nChannels
                for j=1:self.nSlices
                    self.rbcInstancesSlicesSeparate{i,j} = combineIMDS(self.rbcInstanceImds(:,i,j));
                    self.mergedMetadataSlicesSeparate{i,j} = self.combineTables(self.instanceMetadata(:,i,j));
                    self.rbcInstancesSlicesSeparateAug{i,j} = augmentedImageDatastore(...
                        self.masterClassifier.netImageSize(1:2),...
                        self.rbcInstancesSlicesSeparate{i,j},...
                        'ColorPreprocessing', 'gray2rgb');
                end
                
                self.rbcInstancesMerged{i} = combineIMDS(self.rbcInstanceImds(:,i,:));
                self.mergedMetadata{i} = self.combineTables(self.instanceMetadata(:,i,:));
                self.rbcInstancesMergedAug{i} = augmentedImageDatastore(...
                    self.masterClassifier.netImageSize(1:2), self.rbcInstancesMerged{i},...
                    'ColorPreprocessing', 'gray2rgb');
                
            end
            
            % Write the merged metadata for bounding box and label references
            self.exportMetadata();
            
            self.mergedProbs = [];
            for m=1:self.nDatasets
                self.mergedProbs = cat(1,self.mergedProbs, self.probs{m});
            end
            
        end
        
        function classifiers = trainClassifiers(self, classifiers, datasets, loadMaster, customWeightVector, storeInClass)
            % Train classifiers, either from scratch (default) or retrain
            % provided classifiers, either on the merged datasets stored in
            % this class (default) or on datasets provided in the
            % arguments. Note that if providing datasets, each dataset
            % should correspond with the provided classifer (a loop is used
            % to loop through the classifiers and the ith dataset is given
            % to the ith classifier).
            
            % Inputs:
            % classifiers is a cell array of GoogleNetClassifier objects
            % datasets is a cell array of imageDatastore objects
            % loadMaster is a boolean flag whether to start with the master
            % classifier's network and retrain from there.
            
            % customWeightVector is provided to the normalizeClassWeights
            % method in GoogleNetClassifier, and it multiplies the
            % normalized weights by this additional factor.
            
            if nargin < 6 || isempty(storeInClass)
                storeInClass = true;
            end
            
            if nargin < 5 || isempty(customWeightVector)
                customWeightVector = ones(self.nClasses,1);
            end
            
            if nargin < 4 || isempty(loadMaster)
                loadMaster = false;
            end
            
            if nargin < 3 || isempty(datasets)
                choices = {'rbcInstancesMerged','rbcInstancesHumanOnly'};
                [indx,tf] = listdlg('PromptString', 'Select the image datastore to train with',...
                    'SelectionMode','single','ListString',choices);
                if tf
                    datasets = self.(choices{indx});
                else
                    warndlg('User cancelled selection operation. No datasets were selected')
                    return
                end
            end
            
            % Classifier was not supplied. Create new classifiers with the
            % datasets specified
            if nargin < 2 || isempty(classifiers)
                
                % Train classifiers for each wavelength
                self.classifierOutputDir = cell(self.nChannels,1);
                for i = 1:numel(datasets)
                    self.classifierOutputDir{i} = fullfile(self.todaysPath, ['Trained classifier output ',self.pipeOpts.wavelengthLabels{i}]);
                    if ~exist(self.classifierOutputDir{i},'dir')
                        mkdir(self.classifierOutputDir{i});
                    end
                    classifiers{i} = GoogleNetClassifier(self.classes, datasets{i}, self.classifierOutputDir{i}, self.pipeOpts.splitRatio);
                end
            else
                if ~iscell(classifiers)
                    classifiers = {classifiers};
                end
                
                for i=1:numel(classifiers)
                    classifiers{i}.setInputData(datasets{i});
                end
            end
            
            for i=1:numel(classifiers)
                if self.pipeOpts.trimData
                    classifiers{i}.trimLabels();
                end
                
                classifiers{i}.splitLabels();
                classifiers{i}.enableAugmenter();
                
                if loadMaster
                    classifiers{i}.setNet(self.masterClassifier.net, self.masterClassifier.lgraph);
                end
                
                classifiers{i}.normalizeClassWeights(customWeightVector);
                
                opts = trainingOptions('sgdm', ...
                    'MiniBatchSize',64, ...
                    'MaxEpochs',9, ...
                    'InitialLearnRate',1e-2, ...
                    'Shuffle','every-epoch',...
                    'LearnRateSchedule','piecewise',...
                    'LearnRateDropPeriod', 1, ...
                    'LearnRateDropFactor',0.8, ...
                    'ValidationData', classifiers{i}.augimdsVal, ...
                    'ValidationPatience', inf, ...
                    'ValidationFrequency',20, ...
                    'Verbose',true ,...
                    'Plots','training-progress');
                
                classifiers{i}.setTrainingOptions(opts);
                disp(['Training classifier for channel: ' num2str(i)]);
                classifiers{i}.trainNet();
            end
            
            if storeInClass
                self.trainedClassifier = classifiers;
            end
            
        end
        
        function testClassifiers(self, imds, wvLabels, slLabels)
            % Default behavior: runs the trained classifiers on their
            % respective validation datasets. The resulting confusion
            % matrices are plotted and saved.
            
            % If provided, imds input can either be an imds or a cell array of imds. If
            % the former, it will be wrapped in a 1x1 cell for indexing
            % purposes. If the latter, then the first dimension is assumed
            % to be wavelengths, and the second dimension assumed to be
            % focal slices.
            
            % wvLabels is a cell array of strings indicating the wavelength
            % labels to be used for plotting purposes.
            
            % slLabels is a cell array of strings indicating the slice
            % labels to be used for plotting purposes.
            
            if nargin < 4 || isempty(slLabels)
                slLabels = self.pipeOpts.sliceLabels;
            end
            
            if nargin < 3 || isempty(wvLabels)
                wvLabels = self.pipeOpts.wavelengthLabels;
            end
            
            if nargin < 2 || isempty(imds)
                imds = self.rbcInstancesSlicesSeparate;
            else
                if ~iscell(imds)
                    imds = {imds};
                end
            end
            
            nCh = size(imds,1);
            nSl = size(imds,2);
            
            % Classify each channel/slice for all the datasets. We have the labels for
            % all these (from master classifier) so we can evaluate CM's manually.
            resultsFolder = fullfile(self.todaysPath, self.mergedResultsFolderName);
            
            if ~exist(resultsFolder,'dir')
                mkdir(resultsFolder);
            end
            
            % Initialize filenames, confusion charts, etc
            figFilename = cell(nCh, nSl);
            pngFilename = figFilename;
            self.trainedCMSlicesSeparate = zeros(self.nClasses, self.nClasses, nCh, nSl);
            self.trainedCMMerged = zeros(self.nClasses, self.nClasses, nCh);
            self.trainedYPred = cell(nCh, nSl);
            self.trainedProbs = self.trainedYPred;
            
            for i=1:nCh
                for j=1:nSl
                    
                    disp(['Classifying ' wvLabels{i}, ', slice ',  slLabels{j}]);
                    % Classify each individual slice separately
                    [self.trainedYPred{i,j}, self.trainedProbs{i,j}] = ...
                        self.trainedClassifier{i}.classifyDatastore(...
                        imds{i,j}, self.pipeOpts.classMiniBatchsize);
                    
                    % Produce confusion matrices
                    self.trainedCMSlicesSeparate(:,:,i,j) = confusionmat(imds{i,j}.Labels, self.trainedYPred{i,j});
                    self.trainedClassifier{i}.plotConfusionMatrix(imds{i,j}.Labels, self.trainedYPred{i,j});
                    drawnow();
                    
                    figFilename{i,j} = fullfile(self.todaysPath, ...
                        'Merged Training Results', ['Confusion matrix ',...
                        wvLabels{i}, '_', slLabels{j}, '.fig']);
                    
                    pngFilename{i,j} = fullfile(self.todaysPath, ...
                        'Merged Training Results', ['Confusion matrix ',...
                        wvLabels{i}, '_', slLabels{j}, '.png']);
                    
                    savefig(gcf, figFilename{i,j});
                    saveas(gcf, pngFilename{i,j});
                    
                end
            end
            
        end
        
        function plotFocusColorArray(self, instanceFilename, chInds, saveDir)
            
            % Creates an image montage of a single rbc instance across
            % channels/slices, and optionally saves the result.
            
            % instanceFilename: filename (no path) for the rbc instance
            
            % chInds: vector of channel indices if a channel subset is
            % desired. Must be a valid channel index.
            
            if nargin < 4 || isempty(saveDir)
                saveDir = '';
            end
            
            if nargin < 3 || isempty(chInds)
                chInds = 1:self.nChannels;
            end
            
            imgFNames = cell(prod(self.nChannels, self.nSlices),1);
            fig = gobjects(numel(imgFNames),1);
            [~,fNames,~] = cellfun(@fileparts, self.rbcInstancesSlicesSeparate{1,1}.Files, 'UniformOutput',false);
            ind = find(contains(fNames, instanceFilename));
            
            count = 0;
            if ~isempty(ind)
                for j=1:self.nSlices
                    for i=chInds
                        count = count+1;
                        imgFNames{count} = self.rbcInstancesSlicesSeparate{i,j}.Files{ind};
                    end
                end
                
                fig(count) = figure;
                montage(imgFNames,'Size',[self.nSlices, self.nChannels]);
                drawnow();
                
                if ~isempty(saveDir)
                    saveas(fig(count), fullfile(saveDir, [instanceFilename, '.png']));
                end
            end
        end
        
        function writeSortedInstances(self, outputParams, imds, outputDir, probs, preventDuplicateExports, chInd)
            
            % This method writes RBC instances to disk, into labeled
            % folders. Typically this is used for human annotation. The
            % human will re-sort the images into the correct folders in
            % order to correct for the mistakes the machine classifier
            % made. After that's done, the user should invoke
            % 'updateAfterHumanAnnotation' to incorporate the changes.
            
            % Update 2020/02/13: It became clear that sometimes a user
            % might want to perform multiple rounds of human annotation to
            % improve the labels. In that case, we don't want to duplicate
            % any effort so we need to exclude RBC instances that were
            % already annotated by a human from this export process.
            
            % Inputs:
            % imds:  The image datastore from which to write sorted
            % instances
            
            % outputDir: Directory in which to create labeled folders
            % with images
            
            % outputParams: is a struct with field "method". "method" can be one
            % of three values: "equalizeClasses", "lowestConfidence", or "all". If
            % method is "equalizeClasses", then outputParams.maxSortedRBCsToWrite
            % will be used to determine how many instances will be written to
            % disk.
            
            % If "lowestConfidence" is used, then there must also be a
            % field "thresholdMethod":
            % Either:
            % "percentConfidence", or
            % "percentOfPopulation"
            % Another field must also be present: "threshold",
            % which is a percent value (0-100)
            % indicating that if the master classifier confidence score is
            % below threshold then the cell will be exported.
            
            % probs: confidence scores output by the machine classifier for
            % the given imds. This is an array of size [nInstances,
            % nClasses]
            
            if nargin < 7 || isempty(chInd)
                chInd = self.pipeOpts.masterChannelInd;
            end
            
            if nargin < 6 || isempty(preventDuplicateExports)
                preventDuplicateExports = true;
            end
            
            if nargin < 5 || isempty(probs)
                probs = self.mergedProbs;
            end
            
            if nargin < 4 || isempty(outputDir)
                outputDir = self.sortedMasterInstanceDir;
            end
            
            if nargin < 3 || isempty(imds)
                imds = self.rbcInstancesSlicesSeparate{self.pipeOpts.masterChannelInd,self.pipeOpts.masterSliceInd};
            end
            
            if nargin < 2 || isempty(outputParams)
                outputParams.method = 'equalizeClasses';
            end
            
            % Write the machine-sorted instances to labeled folders for human annotation
            % and create imageDatastores that use those instances, including the
            % metadata that goes along with it.
            nInstances = zeros(numel(self.masterClassifier.classes), 1);
            for i=1:numel(self.masterClassifier.classes)
                nInstances(i) = sum(self.masterClassifier.classes{i} == imds.Labels);
            end
            classInds = cell(self.nClasses,1);
            setCount = zeros(self.nClasses,1);
            pickInds = classInds;
            copyStatus = classInds;
            
            % Loop over each class
            for j=1: numel(classInds)
                classInds{j} = find(imds.Labels == self.masterClassifier.classes{j});
                
                % Check if we are exporting any instances that have already
                % been human-labeled
                if preventDuplicateExports && ~isempty(self.rbcInstancesHumanOnly{chInd})
                    [~,imdsFnames,~] = cellfun(@fileparts, imds.Files(classInds{j}),'UniformOutput', false);
                    [~,humanFnames,~] = cellfun(@fileparts, self.rbcInstancesHumanOnly{chInd}.Files, 'UniformOutput',false);
                    [~,newInds] = setdiff(imdsFnames, humanFnames);
                    classInds{j} = classInds{j}(newInds);
                end
                
                switch outputParams.method
                    
                    % Select a random set from each class to write to disk from the dataset
                    case 'equalizeClasses'
                        setCount(j) = min([nInstances(j), outputParams.maxSortedRBCsToWrite]);
                        pickInds{j} = randperm(nInstances(j), setCount(j));
                        self.sortedExportType = 'Partial';
                        
                        
                        % Export a random set of the lowest confidence
                        % instances from each class
                    case 'lowestConfidence'
                        sortedInds = self.confidenceThreshold(probs(classInds{j},:),...
                            outputParams.threshold, ...
                            outputParams.thresholdMethod);
                        
                        % Determine number of instances to export
                        setCount(j) = min([nInstances(j), outputParams.maxSortedRBCsToWrite, numel(sortedInds)]);
                        
                        % These are the highest 'setCount' confidence
                        % instances
                        pickInds{j} = sortedInds(1:setCount(j));
                        self.sortedExportType = 'Partial';
                        
                    case 'highestConfidence'
                        
                        sortedInds = self.confidenceThreshold(probs(classInds{j},:),...
                            outputParams.threshold, ...
                            outputParams.thresholdMethod);
                        
                        % Determine number of instances to export
                        setCount(j) = min([nInstances(j), outputParams.maxSortedRBCsToWrite, numel(sortedInds)]);
                        
                        % These are the highest 'setCount' confidence
                        % instances
                        pickInds{j} = sortedInds(end:-1:(end-setCount(j)+1));
                        self.sortedExportType = 'Partial';
                        
                    case 'all'
                        setCount(j) = nInstances(j);
                        pickInds{j} = 1:nInstances(j);
                        self.sortedExportType = 'Complete';
                        
                end
                
                
                self.machineSortedDirName{j} = fullfile(outputDir, self.masterClassifier.classes{j});
                if ~exist(self.machineSortedDirName{j}, 'dir')
                    mkdir(self.machineSortedDirName{j});
                end
                
                % Copy all the randomly selected RBC instances into labeled folders
                count = 0;
                fNames = imds.Files;
                for k = 1:numel(pickInds{j})
                    count = count+1;
                    disp([self.masterClassifier.classes{j}, ' ', num2str(count)]);
                    
                    fs = fNames{classInds{j}(pickInds{j}(k))};
                    if contains(fs, '\sl1\')
                        sn = 1;
                    elseif contains(fs,'\sl2\')
                        sn = 2;
                    elseif contains(fs,'\sl3\')
                        sn = 3;
                    elseif contains(fs,'\sl4\')
                        sn = 4;
                    elseif contains(fs,'\sl5\')
                        sn = 5;
                    else
                        error('File path does not contain slice number');
                    end
                    
                    if exist(fs, 'file')
                        [~,fn,e] = fileparts(fs);
                        fd = fullfile(self.machineSortedDirName{j},[cat(2,fn,['_sl',num2str(sn,'%1d')]), e]);
                    end
                    
                    [copyStatus{j}(count), msg] = copyfile(fs,...
                        fd);
                    if ~copyStatus{j}(count)
                        warning('Copy file failed');
                        warning(msg);
                    end
                    
                end
                
            end
            
        end
        
        function sortInds = confidenceThreshold(~, probs, thresholdPercent, method, maxOut)
            % Returns the sorted indices for all the probs that are above the given
            % threshold. The threshold can be specified either by
            % threshold value (percent), by percent of the population.
            n = size(probs,1);
            
            if (thresholdPercent < 0) || (thresholdPercent > 100)
                error('thresholdPercent must be in the range [0,100]')
            end
            
            if nargin < 5 || isempty(maxOut)
                maxOut = n;
            end
            
            switch method
                case 'percentConfidence'
                    sortInds = find(max(probs,[],2) > thresholdPercent/100);
                case 'percentOfPopulation'
                    maxProbs = max(probs,[],2);
                    [~, sortInds] = sort(maxProbs,'ascend');
                    sortInds = sortInds(1: min(maxOut,floor(n*thresholdPercent/100)));
            end
        end
        
        function updateAfterHumanAnnotation(self, pathToHumanImds, assertYes)
            % This method receives as input the path to a directory
            % containing human-annotated cells. The directory must contain
            % sub-folders titled with the class labels, and the instances
            % sorted into the sub-folders.
            
            % After creating an imageDatastore object from this dataset,
            % this method reads and parses all the filenames of the sorted
            % instances. It then tries to match the filenames with the
            % filenames in the existing image datastores stored in this
            % class (rbcInstancesMerged and rbcInstancesSlicesSeparate).
            
            if nargin < 3 || isempty(assertYes)
                assertYes = false;
            end
            
            
            if ~assertYes
                answer = yesNoDialog('Has a human carefully sorted these cells?','Confirm human sorting','No',true);
            else
                answer = true;
            end
            
            if answer
                % Create an imageDatastore from the new human-sorted image
                % directory.
                self.humanSortedImds = imageDatastore(pathToHumanImds,'LabelSource','FolderNames', 'IncludeSubfolders',true);
                
                % Update the merged image datastore labels using the new human-sorted imageDatastore
                [self.rbcInstancesMerged, self.mergedMetadata, self.humanOnlyMetadata] = ...
                    self.updateChannelLabels(self.humanSortedImds, self.rbcInstancesMerged, self.mergedMetadata);
                
                % Update the 'slices separate' image datastore labels using the new human-sorted imageDatastore
                [self.rbcInstancesSlicesSeparate, self.mergedMetadataSlicesSeparate, self.humanOnlyMetadata] = ...
                    self.updateChannelLabels(self.humanSortedImds, self.rbcInstancesSlicesSeparate, self.mergedMetadataSlicesSeparate);
                
                % Update the original rbcInstancesImds
                [self.rbcInstanceImds, ~, ~] = ...
                    self.updateChannelLabels(self.humanSortedImds, self.rbcInstanceImds);
                
                % Create a new image datastore that consists of the entries
                % from rbcInstancesMerged but only the cells that have been
                % human-annotated. This is distinct from humanSortedImds,
                % which:
                % a) Does not include all z-slices
                % b) Might be modified (eg. logarithm or rescaling) in ways
                % that make it easier to sort manually).
                
                [~,humanFnames,~] = cellfun(@fileparts, self.humanSortedImds.Files, 'UniformOutput',false);
                
                for i=1:self.nChannels
                    [~,imdsFnames,~] = cellfun(@fileparts, self.rbcInstancesMerged{i}.Files,'UniformOutput', false);
                    inds = false(numel(imdsFnames),1);
                    for k= 1:numel(humanFnames)
                        inds = inds|strcmp(imdsFnames, humanFnames{k});
                    end
                    
                    % Create the new datastore
                    newDS = imageDatastore(pathToHumanImds,'LabelSource','FolderNames', 'IncludeSubfolders',true);
                    newDS.Files = self.rbcInstancesMerged{i}.Files(inds);
                    newDS.Labels = self.rbcInstancesMerged{i}.Labels(inds);
                    
                    % Combine it with the existing human annotations,
                    % ensuring the entries are unique
                    if numel(self.rbcInstancesHumanOnly) >= i
                        self.rbcInstancesHumanOnly{i} = self.combineIMDS({self.rbcInstancesHumanOnly{i}, newDS}, true);
                    else
                        self.rbcInstancesHumanOnly{i} = newDS;
                    end
                end
                
            end
            
        end
        
        function setNegativeControlsToHealthy(self, rawDatasetIDs)
            % Sets all the labels for a given raw dataset to 'healthy'.
            % This is used in the case where positive control data is
            % provided and classified by the master classifier (or a
            % human!) and there is some finite error rate. This is used to
            % correct those labels, as they are all known to be healthy.
            
            % rawDatasetIDS is a string or cell array of strings corresponding
            % to the datasetIDs of the healthy controls
            
            if ~iscell(rawDatasetIDs)
                rawDatasetIDs = {rawDatasetIDs};
            end
            
            healthyInds = zeros(numel(rawDatasetIDs),1);
            for i=1:numel(rawDatasetIDs)
                healthyInds(i) = find(contains(self.rawDataIDs, rawDatasetIDs{i}),1);
            end
            
            % Create a new label so it doesn't create confusion later. One
            % might think the master classifier performed perfectly on this
            % healthy control!
            
            % This makes a copy of the master labels
            for i = 1:numel(self.instanceMetadata)
                if ~isempty(self.instanceMetadata{i})
                    self.instanceMetadata{i}.masterPredictionCorrectedHealthy = self.instanceMetadata{i}.masterClassPrediction;
                end
            end
            
            for m = 1:numel(rawDatasetIDs)
                
                for i = 1:self.nChannels
                    
                    % First set the labels in the mergedMetadata
                    inds = contains(self.mergedMetadata{i}.datasetID, rawDatasetIDs{m});
                    self.mergedMetadata{i}.HumanLabels(inds) = cellstr('healthy');
                    
                    % Then set the labels in the rbcInstancesMerged
                    inds = contains(self.rbcInstancesMerged{i}.Files, rawDatasetIDs{m});
                    self.rbcInstancesMerged{i}.Labels(inds) = cellstr('healthy');
                    
                    % Next set the labels in the image datastores and the
                    % non-merged metadata
                    for j=1:self.nSlices
                        self.rbcInstanceImds{healthyInds(m),i,j}.Labels = ...
                            categorical(ones(numel(self.rbcInstanceImds{healthyInds(m),i,j}.Labels),1), 1, 'healthy');
                        
                        % Then set the labels in the
                        % rbcInstancesSlicesSeparate
                        inds = contains(self.rbcInstancesSlicesSeparate{i,j}.Files, rawDatasetIDs{m});
                        self.rbcInstancesSlicesSeparate{i,j}.Labels(inds) = cellstr('healthy');
                        
                        self.instanceMetadata{healthyInds(m),i,j}.masterPredictionCorrectedHealthy = ...
                            categorical(ones(numel(self.rbcInstanceImds{healthyInds(m),i,j}.Labels),1), 1, 'healthy');
                        
                        % Add these cells (all slices) to the datastore that only has
                        % human-annotated data:
                        self.rbcInstancesHumanOnly{i} = self.combineIMDS({self.rbcInstancesHumanOnly{i}, ...
                            self.rbcInstanceImds{healthyInds(m),i,j}});
                    end
                end
            end
            
            % Ensure that these cells are added to the
            % humanSortedImds so that future exports of the RBC
            % instances for human sorting do not include these cells:
            self.humanSortedImds = self.combineIMDS({self.humanSortedImds, ...
                self.rbcInstanceImds{healthyInds(m),self.pipeOpts.masterChannelInd, ...
                self.pipeOpts.masterSliceInd}});
            
        end
        
        function exportMetadata(self)
            % Writes metadata tables to disk
            
            for i=1:self.nChannels
                writetable(self.mergedMetadata{i}, fullfile(self.todaysPath, ['mergedMetadata_', self.pipeOpts.wavelengthLabels{i}, '.csv']));
                if ~isempty(self.humanOnlyMetadata)
                    writetable(self.humanOnlyMetadata{i}, fullfile(self.todaysPath, ['humanOnlyMetadata_', self.pipeOpts.wavelengthLabels{i}, '.csv']));
                end
            end
            
        end
        
        function [maxWvStats, maxSlStats, maxOvStats] = ...
                maxConfidenceOutput(self, predictions, confidences)
            
            % This method builds a max confidence classification across
            % wavelengths and/or slices.
            
            % Both inputs are in the form of cell arrays whose dimensions
            % are equal to:
            % Dim 1: nChannels
            % Dim 2: nSlices
            % If there is only one channel, Dim 1 needs to be a
            % singleton (ie. 1 x nSlices)
            
            % Each cell array element {i,j} is itself an array that contains the
            % data for the corresponding {channel, slice}.
            
            % predictions: A cell array as described above. Each
            % element is a categorical vector of classifier predictions.
            % Lengths of all categorical vectors must be equal to nInstances.
            
            % confidences: Similar to predictions, but confidences is a
            % cell array that contains the confidence scores for all the
            % elements in predictions. confidences should have dimensions:
            % Dim 1: Wavelength number
            % Dim 2: Slice number
            % Each element in the cell array must be a matrix of size
            % [nInstances, nClasses]. Each row in this array contains
            % probabilities summing to 1.
            
            % As with predictions, confidences must use these dimension
            % assignments even with singleton dimensions
            
            % Outputs:
            % Common: All outputs are structs with the following fields:
            % probs: confidence score multidimensional array.
            % pred: prediction categorical array
            % inds: indices indicating which channel or slice
            % had the highest confidence score for a
            % given instance.
            
            % maxWvStats: Struct with the above fields, each corresponding
            % to selection of the maximum confidence score across
            % wavelengths.
            % Fields:
            % probs: array of doubles with dimensions [nSl, nInst, nCats].
            % pred: categorical array with dimensions [nSl, nInst]
            % inds: array with dimensinos [nSl, nInst]
            
            % maxSlStats: Struct with the above fields, each corresponding
            % to selection of the maximum confidence score across
            % slices.
            % Fields:
            % probs: array of doubles with dimensions [nCh, nInst, nCats].
            % pred: categorical array with dimensions [nCh, nInst]
            % inds: array with dimensinos [nCh, nInst]
            
            % maxOvStats: Same as the other two, but the maximum is
            % computed over both wavelength and slice.
            % Fields:
            % probs: array of doubles with dimensions [1, nInst, nCats].
            % pred: categorical array with dimensions [1, nInst]
            % inds: array with dimensinos [1, nInst]
            
            if nargin < 3 || isempty(confidences)
                confidences = self.trainedProbs;
            else
                if ~iscell(confidences)
                    confidences = {confidences};
                end
            end
            
            if nargin < 2 || isempty(predictions)
                predictions = self.trainedYPred;
            else
                if ~iscell(predictions)
                    predictions = {predictions};
                end
            end
            
            nCh = size(confidences,1);
            nSl = size(confidences,2);
            nInst = size(confidences{1,1},1);
            
            probsArray = zeros(nInst, self.nClasses, nCh, nSl);
            allPredictions = categorical(zeros(nInst, nCh, nSl));
            
            % Convert the cell arrays into arrays
            for i=1:nCh
                for j=1:nSl
                    probsArray(:,:,i,j) = confidences{i,j};
                    allPredictions(:,i,j) = predictions{i,j};
                end
            end
            
            % maxConfArray is the set of max confidence scores of
            % dimensions [nCh, nSl, nInst]
            maxConfArray =  permute(max(probsArray, [], 2), [2,3,4,1]);
            
            cats = categories(predictions{1,1});
            nCats = numel(cats);
            
            % Initialize structs with properties that hold all the statistics
            maxWvStats.pred = categorical(ones(nSl, nInst), 1:nCats, cats);
            maxWvStats.inds = zeros(nSl, nInst);
            maxWvStats.probs = zeros(nSl, nInst, nCats);
            maxSlStats.pred = categorical(ones(nCh, nInst), 1:nCats, cats);
            maxSlStats.inds = zeros(nCh, nInst);
            maxSlStats.probs = zeros(nCh, nInst, nCats);
            maxOvStats.pred = categorical(ones(nInst,1), 1:nCats, cats);
            maxOvStats.inds = zeros(2,nInst);
            
            % Here we loop over instances and use find, which is slower than logical
            % indexing. But doing it this way because if you don't use find()
            % sometimes more than one index is equal to the max
            % probability. Find() has the option to only return one index
            % so it guarantees you don't get duplication errors.
            for k=1:nInst
                for i = 1:nSl
                    maxWvStats.inds(i,k) = find(maxConfArray(1,:,i,k) == max(maxConfArray(1,:,i,k)), 1);
                    maxWvStats.pred(i,k) = allPredictions(k,maxWvStats.inds(i,k),i);
                    maxWvStats.probs(i,k,:) = probsArray(k,:,maxWvStats.inds(i,k), i);
                end
                for j = 1:nCh
                    maxSlStats.inds(j,k) = find(maxConfArray(1,j,:,k) == max(maxConfArray(1,j,:,k)),  1);
                    maxSlStats.pred(j,k) = allPredictions(k,j,maxSlStats.inds(j,k));
                    maxSlStats.probs(j,k,:) = probsArray(k,:,j,maxSlStats.inds(j,k));
                end
                ovInd = find(maxConfArray(1,:,:,k) == max(max(squeeze(maxConfArray(1,:,:,k)),[],1),[],2), 1);
                tempMap = squeeze(allPredictions(k,:,:));
                maxOvStats.pred(k) = tempMap(ovInd);
                [bestChInd, bestSlInd] = ind2sub([nCh,nSl],ovInd);
                maxOvStats.inds(1,k) = bestChInd;
                maxOvStats.inds(2,k) = bestSlInd;
                maxOvStats.probs(k,:) = probsArray(k,:, bestChInd, bestSlInd);
            end
        end
        
        function addDataset(self, newRawDataPath, hasMDImage)
            
            % This function adds a new dataset into the analysis and
            % invokes all the processing steps up to mergeDatasets.
            
            % Inputs:
            % newDatasetID: A single directory name (character array or
            % string) to be added to the analysis. The same assumptions are
            % made of the datasets as in the case of 'setBaseRawDataPath'.
            %
            % hasMDImage: a boolean value indicating whether the dataset is
            % managed by an MDImage object (true), or whether they are raw
            % images (false).
            
            if nargin < 3 || isempty(hasMDImage)
                hasMDImage = true;
            end
            
            if nargin < 2 || isempty(newRawDataPath)
                error('No dataset provided');
            end
            
            % Check that the provided newDatasetID exists
            if ~exist(newRawDataPath, 'dir')
                error([newRawDataPath, ' does not exist!']);
            else
                
                % Add the directory
                self.rawDataPaths{end+1} =  newRawDataPath;
                
                % Find the name of the bottom-level folder. This is the ID
                % for the dataset
                slashInds = strfind(newRawDataPath,'\');
                self.rawDataIDs{end+1} = newRawDataPath((slashInds(end)+1):end);
                self.nDatasets = numel(self.rawDataIDs);
                %
                % Apply pre-processing as appropriate for the type of
                % dataset
                if hasMDImage
                    self.preprocessMDImages(self.nDatasets);
                else
                    self.preprocessRawImages(self.nDatasets);
                end
                
                self.segmentMaster(self.nDatasets);
                self.processInstances(self.nDatasets);
                self.classifyWithMaster(self.nDatasets);
                self.mergeDatasets();
            end
        end
        
        function removeDataset(self, m, assertion)
            
            % Removes an entire raw dataset from the pipeline. This should
            % be performed with great care and is not reversible. 
            
            % m is the index of the dataset to be removed. Check
            % self.rawDataIDs to ensure the correct index is provided. 
            
            % assertion is a boolean value used to assert this operation
            % should be performed. If not provided, the user will be asked.
            % 
            
            if nargin < 3 || isempty(assertion)
                assertion = false;
            else
                if rem(1,m)
                    error('m must be an integer!');
                end
            end
            
            if ~assertion
                answer = yesNoDialog('Are you sure you want to remove this dataset?','Dataset deletion','No',true);
            end
            if answer || assertion
                
                % inds are the new valid dataset indices
                inds = 1:self.nDatasets;
                if (m <= self.nDatasets)
                    inds(m) = [];
                elseif m > self.nDatasets
                    inds = 1:self.nDatasets;
                end
                
                self.rawDataIDs = self.rawDataIDs(inds);
                self.rawDataPaths = self.rawDataPaths(inds);
                self.nPositions = self.nPositions(inds);
                self.preprocFilenamesCleaned = self.preprocFilenamesCleaned(inds,:,:);
                self.preprocImds = self.preprocImds(inds,:,:);
                self.preprocImdsAug = self.preprocImdsAug(inds,:,:);
                self.preprocOutputDir = self.preprocOutputDir(inds);
                self.validFocusInds = self.validFocusInds(inds);
                self.rbcLabelOutputDir = self.rbcLabelOutputDir(inds);
                self.rbcInstanceDir = self.rbcInstanceDir(inds,:,:);
                self.rbcInstanceImds = self.rbcInstanceImds(inds,:,:);
                self.rbcInstanceAug = self.rbcInstanceAug(inds,:,:);
                self.instanceMetadataRaw = self.instanceMetadataRaw(inds);
                self.masterLabelMatrix = self.masterLabelMatrix(inds);
                self.nDatasets = self.nDatasets - 1;
            end
        end
        
        function removeWavelength(self, indToRemove, assertion)
            
            % This function removes a wavelength from object. This is an
            % irreversible operation and should be done with great care.
            
            if nargin < 3 || isempty(assertion)
                assertion = false;
            else
                if numel(indToRemove) > 1
                    error('Can only remove one wavelength at a time!');
                else
                    if isinteger(indToRemove)||(indToRemove<1)
                        error('indToRemove must be an integer greater than 1!');
                    end
                end
            end
            
            if indToRemove == self.pipeOpts.masterChannelInd
                error('Cannot remove master channel');
            end
            
            if ~assertion
                answer = yesNoDialog('Are you sure you want to remove this wavelength? This is permanent!','Wavelength removal','No',true);
            else
                answer = true;
            end
            
            if answer || assertion
                
                % inds are the new valid wavelengths indices
                inds = 1:self.nChannels;
                if (indToRemove <= self.nChannels)&&(indToRemove >= 1)
                    inds(indToRemove) = [];
                elseif (indToRemove > self.nDatasets)
                    inds = 1:self.nChannels;
                end
                
                self.preprocFilenamesCleaned = self.preprocFilenamesCleaned(:,inds,:);
                self.preprocImds = self.preprocImds(:,inds,:);
                self.preprocImdsAug = self.preprocImdsAug(:,inds,:);
                
                for m = 1:self.nDatasets
                    self.validFocusInds{m} = self.validFocusInds{m}(inds,:);
                    self.instanceMetadataRaw{m} = self.instanceMetadataRaw{m}(inds,:);
                end
                self.rbcInstanceDir = self.rbcInstanceDir(:,inds,:);
                self.rbcInstanceImds = self.rbcInstanceImds(:,inds,:);
                self.rbcInstanceAug = self.rbcInstanceAug(:,inds,:);
                self.nChannels = self.nChannels - 1;
                
                self.pipeOpts.wavelengthLabels = self.pipeOpts.wavelengthLabels(inds);
                self.pipeOpts.nChannels = self.pipeOpts.nChannels - 1;
                self.xlsFullFileName = self.xlsFullFileName(:,inds,:);
                self.rbcInstancesMerged = self.rbcInstancesMerged(inds);
                self.rbcInstancesSlicesSeparate = self.rbcInstancesSlicesSeparate(inds,:);
                self.mergedMetadataSlicesSeparate = self.mergedMetadataSlicesSeparate(inds,:);
                self.rbcInstancesSlicesSeparateAug = self.rbcInstancesSlicesSeparateAug(inds,:);
                self.mergedMetadata = self.mergedMetadata(inds);
                self.rbcInstancesMergedAug = self.rbcInstancesMergedAug(inds);
                self.trainedClassifier = self.trainedClassifier(inds);
                self.trainedYPred = self.trainedYPred(inds,:);
                self.trainedProbs = self.trainedProbs(inds,:);
                %                 self.trainedProbsMerged = self.trainedProbsMerged(inds);
                %                 self.trainedYPredMerged = self.trainedYPredMerged(inds);
                self.trainedCMSlicesSeparate = self.trainedCMSlicesSeparate(:,:,inds,:);
                self.trainedCMMerged = self.trainedCMMerged(:,:,inds);
                
                % Shift the master wavelength by one, if appropriate
                if indToRemove < self.pipeOpts.masterChannelInd
                    self.pipeOpts.masterChannelInd = self.pipeOpts.masterChannelInd - 1;
                end
                
            end
        end
        
    end
    
    methods (Static)
        
        function [updatedMachineSortedImds, updatedMetaData, humanOnlyMetaData] = updateChannelLabels(humanSortedImds, machineSortedImds, metaData)
            
            % Update 2020/01/10: This needs to be compatible with a partial
            % human update. This happens when a subset of the population is
            % exported for human annotation, ie. the bottom 10% confidence
            % of cells. In this case, the human-annotated cells need to be
            % incorporated into the existing imds and metadata.
            
            if numel(humanSortedImds) > 1
                error('Only one human sorted datastore allowed');
            end
            
            % Check if machineSortedImds is a cell array of imds. If not, it should
            % be an imds. Then just for compatibility with indexing, we put
            % it into a 1x1 cell array.
            if ~iscell(machineSortedImds)
                if isa(machineSortedImds,'matlab.io.datastore.ImageDatastore')
                    machineSortedImds = {machineSortedImds};
                else
                    error('machineSortedImds must be either an image datastore or a cell array of image datastores');
                end
            else
                for i=1:numel(machineSortedImds)
                    if ~isa(machineSortedImds{i}, 'matlab.io.datastore.ImageDatastore')
                        error('machineSortedImds must be either an image datastore or a cell array of image datastores');
                    end
                end
            end
            
            % metadata was passed in for updating
            if nargin >=3
                
                doMetadata = true;
                
                % Check if metaData is a cell array. If not, it should be a table
                if ~iscell(metaData)
                    if ~istable(metaData)
                        if ~ischar(metaData)
                            error('metaData must be either a table or a cell array of tables');
                        else
                            try
                                metaData = readtable(metaData);
                                metaData = {metaData};
                            catch E
                                warning('Could not read table!')
                                disp(E);
                                throw(E);
                            end
                        end
                    else
                        metaData = {metaData};
                    end
                else
                    % It's a cell array of tables
                end
            else
                doMetadata = false;
            end
            
            nMachine = numel(machineSortedImds);
            humanLabels = humanSortedImds.Labels;
            humanFiles = humanSortedImds.Files;
            nHumanFiles = numel(humanSortedImds.Files);
            
            % Use temp cell arrays because accessing imageDatastore files is really
            % slow for some reason
            machFiles = cell(nMachine,1);
            newLabels = machFiles;
            
            if doMetadata
                nMeta = numel(metaData);
                metaFiles = cell(nMeta,1);
                newTblLabels = metaFiles;
                humanOnlyMetaData = cell(nMachine,1);
            end
            
            % Initialize new cell arrays of labels
            for i=1:nMachine
                [~,machFiles{i},~] = cellfun(@fileparts, machineSortedImds{i}.Files, 'UniformOutput', false);
                newLabels{i} = machineSortedImds{i}.Labels;
            end
            
            if doMetadata
                for i=1:nMeta
                    [~,metaFiles{i}, ~] = cellfun(@fileparts, metaData{i}.InstanceFilename, 'UniformOutput', false);
                    newTblLabels{i} = cell(size(metaData{i},1),1);
                    humanOnlyMetaData{i} = table();
                end
            end
            
            firstMultipleCopyWarning = true;
            firstMultipleCopyWarning_tbl = true;
            firstMachSortedWarning = true;
            firstMachSortedWarning_tbl = true;
            
            hp = waitbar(0,'Merging new human labels into datasets...');
            
            [~, fName, ~] = cellfun(@fileparts, humanFiles, 'UniformOutput', false);
            
            % Transfer the labels from the human-sorted to the machine-sorted databases
            for j=1:nHumanFiles
                waitbar(j/nHumanFiles, hp);
                
                % In case the set of file names has somehow changed, match the file
                % names when transferring labels
                for i=1:nMachine
                    
                    imdsInds = find(strcmp(machFiles{i}, fName{j}));
                    
                    % Parse cases of number of matching imds file names for each file in the
                    % master
                    if numel(imdsInds) == 1
                        newLabels{i}(imdsInds) = cellstr(humanLabels(j));
                        
                    elseif numel(imdsInds) > 1
                        if firstMultipleCopyWarning
                            disp('More than one matching machine-sorted imds entry has been found! Updating all matching labels');
                            firstMultipleCopyWarning = false;
                        end
                        for k = 1:numel(imdsInds)
                            newLabels{i}(imdsInds(k)) = cellstr(humanLabels(j));
                        end
                    else
                        if firstMachSortedWarning
                            warning('Filename in machine-sorted imds not found!');
                            firstMachSortedWarning = false;
                        end
                    end
                    
                end
                
                if doMetadata
                    for i=1:nMeta
                        
                        tblInds = find(strcmp(metaFiles{i}, fName{j}));
                        
                        % Add to a new table that consists only of the
                        % human-annotated entries
                        tempTbl = metaData{i}(tblInds,:);
                        tempHumanLabels = categorical(ones(numel(tblInds),1), 1, cellstr(humanLabels(j)));
                        tempTbl.HumanLabels = tempHumanLabels;
                        humanOnlyMetaData{i} = vertcat(humanOnlyMetaData{i},tempTbl);
                        
                        % Parse cases for the filenames in the table
                        if numel(tblInds) == 1
                            newTblLabels{i}(tblInds) = cellstr(humanLabels(j));
                        elseif numel(tblInds) > 1
                            if firstMultipleCopyWarning_tbl
                                disp('More than one matching machine-sorted metadata entry has been found! Updating all matching labels');
                                firstMultipleCopyWarning_tbl = false;
                            end
                            for k = 1:numel(tblInds)
                                newTblLabels{i}(tblInds(k)) = cellstr(humanLabels(j));
                            end
                        else
                            if firstMachSortedWarning_tbl
                                warning('Filename in metaData table not found');
                                firstMachSortedWarning_tbl = false;
                            end
                        end
                        
                    end
                end
                
            end
            
            % Transfer all the new labels to the output variables
            updatedMachineSortedImds = machineSortedImds;
            for i=1:nMachine
                updatedMachineSortedImds{i}.Labels = newLabels{i};
            end
            
            if doMetadata
                updatedMetaData = metaData;
                for i=1:nMeta
                    updatedMetaData{i}.HumanLabels = newTblLabels{i};
                end
                
            else
                updatedMetaData = [];
                humanOnlyMetaData = [];
            end
            
            delete(hp);
        end
        
        function imdsCombined = combineIMDS(imdsCellArray, ensureUnique)
            % This function simply creates a new image datastore object that is the
            % combination of multiple individual datastores
            
            % Inputs
            % imdsCellArray: a cell array of imageDatastore objects to be
            % combined
            % ensureUnique: boolean flag indicating whether the
            % imdsCombined should not duplicate entries
            
            if nargin < 2 || isempty(ensureUnique)
                ensureUnique = true;
            end
            
            nDatastores = numel(imdsCellArray);
            fileList = {};
            labels = {};
            for i = 1:nDatastores
                if ~isempty(imdsCellArray{i})
                    fileList = cat(1,fileList, imdsCellArray{i}.Files);
                    labels = cat(1,labels, imdsCellArray{i}.Labels);
                end
            end
            
            if ensureUnique
                [fileList, iA, ~] = unique(fileList,'stable');
                labels = labels(iA);
            end
            
            imdsCombined = imageDatastore(fileList);
            imdsCombined.Labels = labels;
        end
        
        function tableCombined = combineTables(tableCellArray)
            
            % This function simply creates a new table that is the
            % vertically concatenated from multiple source tables stored in a cell
            % array
            nTables = numel(tableCellArray);
            tableCombined = tableCellArray{1};
            
            for i = 2:nTables
                % Check if the new table has all the same categories as the
                % concatenated table
                newIsMissing = setdiff(tableCombined.Properties.VariableNames, ...
                    tableCellArray{i}.Properties.VariableNames);
                oldIsMissing = setdiff(tableCellArray{i}.Properties.VariableNames, ...
                    tableCombined.Properties.VariableNames);
                
                % Add any needed columns to the combined table, populate
                for j=1:numel(oldIsMissing)
                    if isnumeric(tableCellArray{i}.(oldIsMissing{j}))
                        newCol = nan(size(tableCombined,1),1);
                    elseif isa(tableCellArray{i}.(oldIsMissing{j}), 'categorical')
                        newCol = categorical(ones(size(tableCombined,1),1), 1, 'NaN');
                    end
                    tableCombined = cat(2,tableCombined, array2table(newCol, 'VariableNames', oldIsMissing(j)));
                end
                
                % Add any needed comlumns to the incoming table
                for j=1:numel(newIsMissing)
                    if isnumeric(tableCombined.(newIsMissing{j}))
                        newCol = nan(size(tableCellArray{i},1),1);
                    elseif isa(tableCombined.(newIsMissing{j}), 'categorical')
                        newCol = categorical(ones(size(tableCellArray{i},1),1), 1, 'NaN');
                    end
                    tableCellArray{i} = cat(2,tableCellArray{i}, array2table(newCol, 'VariableNames', newIsMissing(j)));
                end
                
                % Add the incoming table to the combined table;
                tableCombined = cat(1,tableCombined, tableCellArray{i});
            end
            
        end
        
        function img = convertTo16bit(filename)
            % Helper function to convert images to 16-bit.
            % Data is scaled using conservative histogram-based upper and
            % lower-limit finding.
            
            % Average over color channels if present
            img = mean(double(imread(filename)),3);
            
            % Use histogram-based rescaling to be robust against hot pixels
            [a,xout] = hist(img(:),10000);
            csum = cumsum(a);
            csum = csum/max(csum);
            maxInd = find(csum > .999,1);
            ma = xout(maxInd);
            minInd = find(csum > .001,1);
            mi = xout(minInd);
            
            img = uint16(65535*(double(img)-mi)/(ma-mi));
        end
        
        function img = convertTo8bit(filename)
            % Helper function to convert images to 8-bit.
            % Data is scaled using conservative histogram-based upper and
            % lower-limit finding.
            
            % Average over color channels if present
            img = mean(double(imread(filename)),3);
            
            % Use histogram-based rescaling to be robust against hot pixels
            [a,xout] = hist(img(:),10000);
            csum = cumsum(a);
            csum = csum/max(csum);
            maxInd = find(csum > .999,1);
            ma = xout(maxInd);
            minInd = find(csum > .001,1);
            mi = xout(minInd);
            
            img = uint8(255*(double(img)-mi)/(ma-mi));
        end
        
    end
end