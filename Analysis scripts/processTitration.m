% processTitration is a script that uses MetaRBCPipeline to process a test
% experiment where the parasitemia in a sample is diluted by serial
% dilution into healthy RBCs.

% Author: Paul Lebel
% czbiohub
% 2020/03/12

%% 1. Set user-defined variables
pipelineOptionsPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\UVScope\Titration 2020-06-20\metaPipelineSettings - titration.json';
baseRawDataPath = '\\ess2.czbiohub.org\flexo\MicroscopyData\Bioengineering\UV Microscopy\RawData\Titration 2020-06-20\';
baseOutputPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\UVScope\Titration 2020-06-20\';

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
fg.OldCats = {'healthy','ring','troph','schizont'};
fg.NewCats = fg.OldCats;
cg.OldCats = {{'troph', 'schizont'}};
cg.NewCats = {'late'};
cg2.OldCats = {{'ring','troph','schizont'}};
cg2.NewCats = {'parasitized'};

%% Define the MetaPipeline object and pre-process data
myMeta = MetaRBCPipeline();
myMeta.setBaseOutputPath(baseOutputPath);
myMeta.setBaseRawDataPath(baseRawDataPath);
myMeta.loadPipelineOptions(pipelineOptionsPath);
myMeta.loadMasterSegmenter();
myMeta.loadMasterClassifier();
myMeta.preprocessMDImages(); %
myMeta.segmentMaster();
myMeta.processInstances();
myMeta.classifyWithMaster();
myMeta.mergeDatasets();

%% Post processing

% Define all the points in the titration, using the lumped manual high
% precision count as the top value.
c_titration = 18.56.*0.5.^(0:9);

% Maps the individual datasets to the titration points.
cMapInds = [1,2,3,4,5,6,8,7,9,10];

% Classify the separate datasets by slice
for m = 1:numel(cMapInds)
    for j=1:myMeta.nSlices
        [mcSliceSepPred{m,j}, mcSliceSepProbs{m,j}] = ...
            myMeta.masterClassifier.classifyDatastore(...
            myMeta.rbcInstanceImds{m,1,j});
    end
    
    % Do max confidence voting on slices
    [~, mcMaxSlStats{m}, ~] = myMeta.maxConfidenceOutput(mcSliceSepPred(m,:), mcSliceSepProbs(m,:));
    mcMaxSlPred{m} = mcMaxSlStats{m}.pred;
end

%% Plotting
% Define a vector of sample total cell counts in order to define the error
% contours associated with counting that many cells
cVec = linspace(c_titration(1), c_titration(end), 500);
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);

for j=1:myMeta.nSlices
    statsOutUnique(j) = mergeConcentrations(c_titration, cMapInds, mcSliceSepPred(:,j));
%     loglog(c_titration, statsOutUnique(j).parasitemiaPercent, 'o-b'); hold all;
end

sliceMean =  mean([statsOutUnique.parasitemiaPercent], 2);
loglog(c_titration, sliceMean, 'o-r','markersize',12,'linewidth',2);

legStr = {'1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)',...
    '1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
    '1\sigma (N = 100000)','Raw classifier','Slice Max Confidence','Human QC'};

grid on;
axis square;
xlim([0.034, 30])
ylim([0.01, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
legend(legStr,'location','southeast');
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
title({'UV Scope 285 nm classifier performance'},'fontsize',16);
%% Merge slice max confidence data
statsOutMaxConf = mergeConcentrations(c_titration, cMapInds, mcMaxSlPred);

hold all;
loglog(c_titration, statsOutMaxConf.parasitemiaPercent,'s-','linewidth',2,'markersize',12);
legStr{end+1} = 'Slice Max Confidence';

%% Manual confidence threshold
confThres = [0.6:.02:.98];
[R,F] = ndgrid(linspace(.3,1,400), linspace(0,0.5,400));
sliceInd = 3;
jetMap = jet();
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);
hold all;

legStr = {'1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)',...
    '1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
    '1\sigma (N = 100000)'};

clear M recall FPR nCells;


color1 = [247,192,10];
color2 = [20,41,200];
colorSpace = flipud([linspace(color1(1), color2(1), numel(confThres))', ...
    linspace(color1(2), color2(2), numel(confThres))', ...
    linspace(color1(3), color2(3), numel(confThres))']/255); 

for k = 1:numel(confThres)
    
    
    for m=1:numel(c_titration)
        [pred, probs] = myMeta.masterClassifier.mergeCategories(fg, myMeta.masterClassPrediction(m),  myMeta.probs{m});
        probs = probs{1};
        pred = pred{1};
        inds = find( max(probs,[],2) >= confThres(k));
        predThres{m,k} = pred(inds);
    end
        
    statsOutThres{k} = mergeConcentrations(c_titration, cMapInds, predThres(:,k));

    nCells(k) = sum(statsOutThres{k}.nCells);
        
    % Compute each curve's optimal recall and FPR compensation
    for i = 1:size(F,1)
        for j= 1:size(F,2)
            M(i,j,k) = sqrt(mean(((c_titration/100).*statsOutThres{k}.nCells').*(((statsOutThres{k}.parasitemiaPercent' - F(i,j))/R(i,j))./c_titration - 1).^2));
        end
    end
    m = M(:,:,k);
    bestInd(k) = find(m == min(m(:)),1);
    recall(k) = R(bestInd(k));
    FPR(k) = F(bestInd(k));
    
    loglog(c_titration, (statsOutThres{k}.parasitemiaPercent - FPR(k))/recall(k),'o-','color', colorSpace(k,:),'linewidth',2); hold all;
    legStr{end+1} = ['Threshold = ', num2str(100*confThres(k)), ' %'];
end

grid on;
axis square;
xlim([0.034, 30])
ylim([0.01, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
% legend(legStr,'location','southeast');
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
title({'285 nm Confidence thresholding','Optimal Recall and FPR compensation'},'fontsize',16);

figure('Position',[50,50,300,700]);
subplot(3,1,1)
plot(confThres, recall*100,'-k','linewidth',2);
title('Statistics vs. Threshold');
ylabel('Recall (%)','fontsize',14);
xlim([min(confThres), max(confThres)]);
subplot(3,1,2)
plot(confThres, FPR,'-k','linewidth',2);
ylabel('False Positive Rate (%)','fontsize',14);
xlim([min(confThres), max(confThres)]);
subplot(3,1,3);
plot(confThres,100* nCells/nCells(1),'-k','linewidth',2);
xlabel('Confidence Threshold (%)','fontsize',16)
ylabel('Cells remaining (%)','fontsize',16);
xlim([min(confThres), max(confThres)]);

%% Human error correction
myMeta.writeSortedInstances(rbcWriteParams, [],[],[], true);

%% Human corrects errors before running this
myMeta.updateAfterHumanAnnotation();

%% Go through datasets and tally results
for m = 1:myMeta.nDatasets
    humanErrCorrPred{m} = myMeta.rbcInstanceImds{m,1,sliceInd}.Labels;
end
statsOutErrCorr = mergeConcentrations(c_titration, cMapInds, humanErrCorrPred);

loglog(c_titration, statsOutErrCorr.parasitemiaPercent, 'h-', 'linewidth',2,'markersize',12); 
legStr{end+1} = 'Human collaboration';

%% Make Giemsa stain plots
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);
grid on;
axis square;
xlim([0.034, 30])
ylim([0.01, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
legend(legStr,'location','southeast');
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
loglog(c_titration, rebCounts(:,5),'--o','linewidth',2,'markersize',12)
loglog(c_titration, valCounts(:,5),'--h','linewidth',2,'markersize',12)
loglog(c_titration, madhuraCounts(:,5),'--p','linewidth',2,'markersize',12)

humanMean(:,1:4) = valCounts(:,1:4) + rebCounts(:,1:4) + madhuraCounts(:,1:4);
humanMean(:,5) = sum(humanMean(:,2:4),2)*100./sum(humanMean(:,1:4),2);
loglog(c_titration, humanMean(:,5),'b-o','linewidth',2,'markersize',12)
legStr = {'1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)',...
    '1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
    '1\sigma (N = 100000)','Human 1','Human 2','Human 3', 'Human Average'};
legend(legStr);

%% Make error plots

% Giemsa
figure('Position',[50,50,700,300]);
semilogx(c_titration, rebCounts(:,5)./c_titration', '-ks','markersize',12);
hold all;
semilogx(c_titration, valCounts(:,5)./c_titration', '-kp','markersize',12);
semilogx(c_titration, madhuraCounts(:,5)./c_titration', '-kh','markersize',12);
semilogx(c_titration, humanMean(:,5)./c_titration', '-bo','linewidth',2,'markersize',12);

grid on;
xlim([0.034, 30])
ylim([-0.5, 9]);
set(gca,'YTick',[0:2:9]);
xlabel('Nominal Parasitemia (%)','fontsize',12);
ylabel('Ratiometric error','fontsize',12);
legend('Human 1','Human 2','Human 3','Human (Average)');
title('Ratiometric counting error','fontsize',16);

% UV Scope
figure('Position',[50,50,700,300]);
semilogx(c_titration, sliceMean./c_titration', '-o','markersize',12, 'linewidth',2);
hold all;
semilogx(c_titration, statsOutMaxConf.parasitemiaPercent./c_titration', '-s','markersize',12, 'linewidth',2);
semilogx(c_titration, statsOutErrCorr.parasitemiaPercent./c_titration', '-h','markersize',12, 'linewidth',2);

grid on;
xlim([0.034, 30])
ylim([-0.5, 9]);
set(gca,'YTick',[0:2:9]);
xlabel('Nominal Parasitemia (%)','fontsize',12);
ylabel('Ratiometric error','fontsize',12);
legend('Raw classifier','Slice Max Confidence','Human QC');
title('Ratiometric counting error','fontsize',16);