


% Requirements:
% Matlab 2019b
% Computer Vision Toolbox
% Deep Learning Toolbox
% Deep Learning Toolbox Model for ResNet-18 Network

classdef WholeBloodSegmenter < SemanticSegmenter
    
    
    properties(Access = public)
        areaConstraints
        solidityConstraints
        circularityConstraints
        tileCanvasSize = [120,120];
    end
    
    properties(Access = private)
        
        
        % Define the colormap for display
        cmap = [...
            000 000 000; ... % "Background"
            230 080 080; ... % "RBC"
            200 200 200; ... % "Lymphocyte"
            080 230 080; ... % "Platelet"
            200 040 040; ... % "CrenulatedRBC"
            200 000 200; ... % "Merozoite"
            064 128 192; ... % "Ring"
            192 128 192; ... % "Troph"
            192 064 128; ... % "Schizont"
            128 064 064; ... % "RupturedSchizont"
            ]./255;
        
        areaRanges = {
            [0 inf], ... % "Background"
            [2000 12000]}; % "RBC"
        %             [0 inf], ... % "Lymphocyte"
        %             [0 inf], ... % "Platelet"
        %             [0 inf], ... % "CrenulatedRBC"
        %             [0 inf], ... % "Merozoite"
        %             [0 inf], ... % "Ring"
        %             [0 inf], ... % "Troph"
        %             [0 inf], ... % "Schizont"
        %             [0 inf], ... % "RupturedSchizont"
        %             };
        
        circularityRanges = {
            [0 1], ... % "Background"
            [0.85 inf]}; % "RBC"
        %             [0 1], ... % "Lymphocyte"
        %             [0 1], ... % "Platelet"
        %             [0 1], ... % "CrenulatedRBC"
        %             [0 1], ... % "Merozoite"
        %             [0 1], ... % "Ring"
        %             [0 1], ... % "Troph"
        %             [0 1], ... % "Schizont"
        %             [0 1], ... % "RupturedSchizont"
        %             };
        
        solidityRanges = {
            [0 1], ... % "Background"
            [0.95 1]}; % "RBC"
        %             [0 1], ... % "Lymphocyte"
        %             [0 1], ... % "Platelet"
        %             [0 1], ... % "CrenulatedRBC"
        %             [0 1], ... % "Merozoite"
        %             [0 1], ... % "Ring"
        %             [0 1], ... % "Troph"
        %             [0 1], ... % "Schizont"
        %             [0 1], ... % "RupturedSchizont"
        %             };
        
        maxRBCShift = 10;
        
    end
    
    methods(Access = public)
        
        function self = WholeBloodSegmenter(imgSize, netName)
            
            if nargin < 2
                netName = 'resnet50';
            end
            
            classes = {'Background', 'RBC'};
            %                 'Lymphocyte', 'Platelet', 'CrenulatedRBC', ...
            %                 'Merozoite', 'Ring', 'Troph', 'Schizont', 'RupturedSchizont'};
            
            % These are the pixel values that define classes in the
            % annotation data.
            
            % NOTE: These are RGB values only because the network we are
            % re-training is set up to use color data. For our purposes, we
            % are only using single-color data, and only use monochromatic
            % labelIDs.
            labelIDs = {
                0, ... % "Background"
                255}; %, ... % "RBC"
            %                 3, ... % "Lymphocyte"
            %                 4, ... % "Platelet"
            %                 5, ... % "CrenulatedRBC"
            %                 6, ... % "Merozoite"
            %                 7, ... % "Ring"
            %                 8, ... % "Troph"
            %                 9, ... % "Schizont"
            %                 10, ... % "RupturedSchizont"
            %                 };
            
            self = self@SemanticSegmenter(netName, classes, labelIDs, imgSize);
            
            self.areaConstraints = containers.Map(self.classes, self.areaRanges);
            self.solidityConstraints = containers.Map(self.classes, self.solidityRanges);
            self.circularityConstraints = containers.Map(self.classes, self.circularityRanges);
        end
        
        function setTileCanvasSize(self, tileCanvasSize)
            % tileCanvasSize: two-element vector
            self.tileCanvasSize = tileCanvasSize;
        end
        
        function setAreaRange(self, className, areaRange)
            % Class is a string matching the class name
            % areaRange is a two-element vector [min, max]
            self.areaConstraints(className) = areaRange;
        end
        
        function imgWS = watershedSeparation(self, bwImg, classID)
            
            % Remove small specks
            areaRange = self.areaConstraints(classID);
            bwImg = bwareaopen(bwImg, areaRange(1));
            
            % Fill holes
            bwImg = imfill(bwImg, 'holes');
            
            % Compute distance map to background pixels, then invert
            D = bwdist(~bwImg);
            D = max(D(:)) - D;
            
            %             D = imgaussfilt(D, filtVal);
            D = medfilt2(D);
            mask = imextendedmin(D,1);
            
            % Check for NaNs, which happens if the input image is just
            % zeros.
            if any(isnan(D))
                warning('Warning! NaN found prior to watershed step! Setting NaNs to zero');
                D(isnan(D)) = 0;
            end
            
            D2 = imimposemin(D,mask);
            Ld2 = watershed(D2);
            imgWS = bwImg;
            imgWS(Ld2 == 0) = 0;
        end
        
        function [tiles, instanceMetadata] = processInstances(self, imdsLabels, imdsOriginals, outputPaths, optionsIn)
            % This method is for post-processing the images that have
            % undergone semantic segmentation. The algorithm is:
            % - Create binary image from each labelID
            % - Clean up and watershed segment the result in order to
            % separate instances
            % - Filter them by geometrical properties, ie.
            % size/roundness/solidity/etc
            % - Crop them with bounding boxes
            % - Center them on new blank canvasses
            % - Export them to folders
            
            % Inputs:
            % className: (String) name of the class that is being instanced
            
            % imdsLabels: an imageDatastore object containing the label
            % matrices that resulted from image segmentation
            
            % imdsOriginals: This is the dataset to which
            % the instance binary masks are applied and exported. One can
            % pass an array of imdsOriginals, to which the same
            % segmentation and instanciation will be applied
            
            % outputPath: Director(ies) where the class instances will be
            % saved
            
            fieldNames = {'className', 'masterIndex','labelMap',...
                'writeRBCInstances','dataTypeOut','excludeEdges','filePrefix'};
            defaults = {'RBC',1,self.labelMap, true, 'uint16',true,''};
            
            if nargin < 5
                optionsIn = struct();
            end
            
            for i=1:numel(fieldNames)
                if isfield(optionsIn, fieldNames{i})
                    options.(fieldNames{i}) = optionsIn.(fieldNames{i});
                else
                    options.(fieldNames{i}) = defaults{i};
                end
            end
            
            % Remove slashes from filePrefix
            tempInd = strfind(options.filePrefix,'\');
            options.filePrefix(tempInd) = '';
            tempInd = strfind(options.filePrefix,'/');
            options.filePrefix(tempInd) = '';
            
            if nargin < 4 || isempty(outputPaths)
                outputPaths = self.outputDir;
            end
            if nargin < 3 || isempty(imdsOriginals)
                imdsOriginals = self.imds;
            end
            if nargin < 2 || isempty(imdsLabels)
                imdsLabels = self.segResults;
            end
            
            if numel(outputPaths) ~= numel(imdsOriginals)
                error('numel(imdsOriginal) must equal numel(outputPaths)');
            end
            
            
            switch options.dataTypeOut
                case 'uint8'
                    outputBitDepth = 8;
                case 'uint16'
                    outputBitDepth = 16;
            end
            
            nOrigImds = numel(imdsOriginals);
            
            % Initialize arrays for saving into .xls format later
            bBox_xls{nOrigImds} = [];
            fName_xls = bBox_xls;
            dSet_xls = bBox_xls;
            imgIndex_xls = bBox_xls;
            rbcIndex_xls = bBox_xls;
            parentFname_xls = bBox_xls;
            
            tileSize = self.tileCanvasSize;
            
            tileFig = figure();
            nImagesPerDataset = numel(imdsOriginals{1}.Files);
            tempImg = squeeze(mean(imread(imdsOriginals{1}.Files{1}),3));
            origDims = size(tempImg);
            nFiles = numel(imdsOriginals{1}.Files);
            tiles = cell(nFiles,1);
            % Loop through each file in the imds
            for i=1:nFiles
                disp(['Processing instances from image: ', num2str(i) '/' num2str(nImagesPerDataset)]);
                labImgIn = imread(imdsLabels.Files{i});
                
                % Select pixels just from this class
                labelVal = mean(options.labelMap(options.className));
                binaryImg = (labImgIn == labelVal(1));
                labelImg = self.watershedSeparation(binaryImg, options.className);
                
                R = regionprops(labelImg, 'Centroid', 'FilledArea', ...
                    'BoundingBox','Circularity','Eccentricity','Solidity',...
                    'Image');
                
                % Pull shape constraints for the given class
                areaRange = self.areaConstraints(options.className);
                circRange = self.circularityConstraints(options.className);
                solidRange = self.solidityConstraints(options.className);
                
                % Configure whether ROIs touches edges will be included or
                % not
                if options.excludeEdges
                    bbLL = 1;
                    bbULx = origDims(2);
                    bbULy = origDims(1);
                else
                    bbLL = -inf;
                    bbULx = inf;
                    bbULy = inf;
                end
                
                bb = cat(1,R.BoundingBox);
                
                % Apply shape conditions
                inds = find([R.FilledArea] > areaRange(1) & ...
                    [R.FilledArea] < areaRange(2) & ...
                    [R.Circularity] > circRange(1) & ...
                    [R.Solidity] > solidRange(1) & ...
                    [R.Solidity] < solidRange(2) & ...
                    bb(:,1)' > bbLL & ...
                    bb(:,2)' > bbLL & ...
                    (bb(:,1)' + bb(:,3)') < bbULx & ...
                    (bb(:,2)' + bb(:,4)') < bbULy);
                
                % Populate tiles with segmented instances, and save
                % them to disk
                tileCount = 0;
                tiles{i} = (2^outputBitDepth-1)*ones([tileSize(1), tileSize(2), nOrigImds, numel(inds)], options.dataTypeOut);
                
                origImages = cell(nOrigImds,1);
                
                ax = gobjects(nOrigImds,1);
                imgHnd = gobjects(nOrigImds);
                % Read in the original imds images
                for j = 1:nOrigImds
                    origImages{j} = squeeze(mean(imread(imdsOriginals{j}.Files{i}),3));
                    % Crop first and last row and column from original images,
                    % which seems to commonly have a dim border that affects
                    % cells at the edge of the images
                    origImages{j}(1,:,:) = (2^outputBitDepth) - 1;
                    origImages{j}(end,:,:) = (2^outputBitDepth) - 1;
                    origImages{j}(:,1,:) = (2^outputBitDepth) - 1;
                    origImages{j}(:,end,:) = (2^outputBitDepth) - 1;
                    
                    if nOrigImds <= 5
                        ax(j)= subplot(1,nOrigImds,j);
                    else
                        ax(j) = subplot(ceil(sqrt(nOrigImds)), ceil(sqrt(nOrigImds)), j);
                    end
                    if iscolumn(tileSize)
                        tileSize = tileSize';
                    end
                    imgHnd(j) = imagesc(ax(j), zeros(tileSize)); colormap gray; axis image;
                end
                
                origDims = size(origImages{j});
                
                % k is Each RBC instance
                for k=inds
                    tileCount = tileCount+1;
                    
                    %% Shift the ROI if the RBC moved between collection of different channels
                    % Centroid of binary region of the master label matrix
                    centerCoords = round(R(k).Centroid);
                    
                    % Crop out a tile centered on that original centroid
                    % centerCoords are (x,y) as opposed to normal row,col
                    indsX = (1:self.tileCanvasSize(2)) + centerCoords(1) - round(self.tileCanvasSize(2)/2);
                    indsY = (1:self.tileCanvasSize(1)) + centerCoords(2) - round(self.tileCanvasSize(1)/2);
                    indsX = max(min(indsX, origDims(2)),1);
                    indsY = max(min(indsY, origDims(1)),1);
                    origCrop = zeros(numel(indsY), numel(indsX), nOrigImds);
                    for j=1:nOrigImds
                        origCrop(:,:,j) = origImages{j}(indsY, indsX,:);
                    end
                    % Align the gradient magnitude of the image
                    if nOrigImds == 1
                        %                         [gy,gx] = gradient(origCrop);
                        dx = 0;
                        dy = 0;
                    else
                        [gy,gx] = gradient(origCrop);
                        origEdges = sqrt(gy.^2 + gx.^2);
                        
                        % [dx, dy] is the amount of translation that happened
                        % between channels for this particular RBC
                        [dx, dy, alignedCells] = imageCrossCorr(origEdges(:,:,options.masterIndex), origEdges, false, false, false, true);
                        for j = 1:nOrigImds
                            if (abs(dx(j)) > self.maxRBCShift) || (abs(dy(j))  > self.maxRBCShift)
                                dx(j) = 0;
                                dy(j) = 0;
                            end
                        end
                    end
                    
                    
                    %                     dx = min(max(-self.maxRBCShift, dx), self.maxRBCShift);
                    %                     dy = min(max(-self.maxRBCShift, dy), self.maxRBCShift);
                    
                    if iscolumn(self.tileCanvasSize)
                        self.tileCanvasSize = self.tileCanvasSize';
                    end
                    mask = zeros(self.tileCanvasSize,'logical');
                    maskDims = size(R(k).Image);
                    centerIndsY = round((self.tileCanvasSize(1)-maskDims(1))/2) + (1:maskDims(1));
                    centerIndsX = round((self.tileCanvasSize(2)-maskDims(2))/2) + (1:maskDims(2));
                    centerIndsX = max(min(centerIndsX, tileSize(2)),1);
                    centerIndsY = max(min(centerIndsY, tileSize(1)),1);
                    mask(centerIndsY,centerIndsX) = R(k).Image;
                    
                    % Dilate the mask in order to see the cell boundaries
                    % in the tiles
                    mask = imdilate(mask, ones(5,5));
                    fgInds = (mask(:) ~= 0);
                    bgInds = (mask(:) == 0);
                    
                    for j = 1:nOrigImds
                        
                        % Shift the original image into place
                        tempTile = mask.*imtranslate(origCrop(:,:,j), [-dx(j),-dy(j)],'FillValues',(2^outputBitDepth)-1);
                        
                        % Use histogram-based rescaling to be robust against hot pixels
                        [a,xout] = hist(tempTile(fgInds),10000);
                        csum = cumsum(a);
                        csum = csum/max(csum);
                        maxInd = find(csum > .999,1);
                        ma = xout(maxInd);
                        minInd = find(csum > .001,1);
                        mi = xout(minInd);
                        
                        % Rescale dynamic range of RBC pixels
                        tempTile(fgInds) = ((2^outputBitDepth)-1)*(tempTile(fgInds)-mi)/(ma-mi);
                        
                        % Set background to white
                        tempTile(bgInds) = (2^outputBitDepth)-1;
                        alignedCells(:,:,j) = tempTile;
                        tiles{i}(:,:,j, tileCount) = alignedCells(:,:,j);
                        imgHnd(j).CData = alignedCells(:,:,j);
                        imageFname = string([options.filePrefix, '_img_' num2str(i,'%03i'), '_', options.className, '_' num2str(tileCount, '%05i')  '.tif']);
                        [~,parentFilename,~] = fileparts(imdsOriginals{j}.Files{i});
                        parentFilename = string(parentFilename); 
                        parentFname_xls{j} = cat(1,parentFname_xls{j}, parentFilename);
                        dSet_xls{j} = cat(1, dSet_xls{j}, string(options.filePrefix));
                        fName_xls{j} = cat(1, fName_xls{j}, imageFname);
                        imgIndex_xls{j} = cat(1,imgIndex_xls{j},i);
                        bBox_xls{j} =  cat(1,bBox_xls{j}, round([R(k).BoundingBox] + [dx(j),dy(j),0,0]));
                        rbcIndex_xls{j} = cat(1,rbcIndex_xls{j}, tileCount);
                        
                        % Save images to disk
                        if options.writeRBCInstances
                            switch options.dataTypeOut
                                case 'uint8'
                                    imwrite(uint8(alignedCells(:,:,j)), fullfile(outputPaths{j}, imageFname));
                                case 'uint16'
                                    imwrite(uint16(alignedCells(:,:,j)), fullfile(outputPaths{j}, imageFname));
                            end
                        end
                    end
                    drawnow();
                end
            end
            
            delete(tileFig);
            
            for j=1:nOrigImds
                instanceMetadata{j} = table();
                instanceMetadata{j}.InstanceFilename = fName_xls{j};
                instanceMetadata{j}.ParentFilename = parentFname_xls{j};
                instanceMetadata{j}.datasetID = dSet_xls{j};
                instanceMetadata{j}.imageIndex = imgIndex_xls{j};
                instanceMetadata{j}.RBCIndex = rbcIndex_xls{j};
                instanceMetadata{j}.BB_x = bBox_xls{j}(:,1);
                instanceMetadata{j}.BB_y = bBox_xls{j}(:,2);
                instanceMetadata{j}.BB_w = bBox_xls{j}(:,3);
                instanceMetadata{j}.BB_h = bBox_xls{j}(:,4);
            end
        end
    end
end