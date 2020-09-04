% processTitration is a script that uses MetaRBCPipeline to process a test
% experiment where the parasitemia in a sample is diluted by serial
% dilution into healthy RBCs. 

% Author: Paul Lebel
% czbiohub
% 2020/03/12

%% 1. Set user-defined variables
pipelineOptionsPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\Leica scope\Titration 2020-06-20\pipelineOptions.json';
baseRawDataPath = 'C:\Users\SingleCellPicker\Documents\Temp data\SCP-2020-06-20 Titration\';
baseOutputPath = 'C:\Users\SingleCellPicker\Documents\Temp data analysis\Leica scope\Titration 2020-06-20\';

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
myMeta.loadPipelineOptions(pipelineOptionsPath);
myMeta.setBaseRawDataPath(baseRawDataPath);
myMeta.loadMasterSegmenter();
myMeta.loadMasterClassifier();
myMeta.preprocessRawImages(); % 
myMeta.segmentMaster();
myMeta.processInstances();
myMeta.classifyWithMaster();
myMeta.mergeDatasets();

%% Post processing

% Define all the points in the titration
c_titration = 18.56.*0.5.^(0:9);
FPR = 0.207;
recall = 0.7172;

% Maps the individual datasets to the titration points. Sometimes more than
% one dataset was collected per titration point.
cMapInds = [1,10,2:9];

clear pred;
for m=1:10
    [temp, ~] = myMeta.masterClassifier.mergeCategories(cg2, myMeta.masterClassPrediction(m),  myMeta.probs{m});
    pred{m} = temp{1};
end
statsOut = mergeConcentrations(c_titration, cMapInds, pred);

% Define a vector of sample total cell counts in order to define the error
% contours associated with counting that many cells
cVec = linspace(c_titration(1), c_titration(end), 500);
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);

hold all;
plot(c_titration, statsOut.parasitemiaPercent, 'go-','linewidth',2,'markersize',12);
plot(c_titration, statsOut.parasitemiaPercent/recall - FPR,'g*-','linewidth',2,'markersize',12);
set(gca, 'XScale','log', 'YScale','log')

legStr = {'1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)',...
    '1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
    '1\sigma (N = 100000)','Raw classifier','Recall and FPR compensated','Human QC'};

grid on;
axis square;
xlim([0.034, 30])
ylim([0.01, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
legend(legStr,'location','southeast');
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
title({'Leica 405 nm classifier performance'},'fontsize',16);


%% Manual confidence threshold
confThres = [0.6:.02:0.98];
[R,F] = ndgrid(linspace(.4,0.8,400), linspace(0.04,0.4,400));
sliceInd = 1;
jetMap = jet();
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);
hold all;
legStr = {'1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)',...
    '1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
    '1\sigma (N = 100000)'};

color1 = [247,192,10];
color2 = [20,41,200];
colorSpace = flipud([linspace(color1(1), color2(1), numel(confThres))', ...
    linspace(color1(2), color2(2), numel(confThres))', ...
    linspace(color1(3), color2(3), numel(confThres))']/255); 


clear M recall FPR nCells;

for k = 1:numel(confThres)
    
    
    for m=1:numel(c_titration)
        [pred, probs] = myMeta.masterClassifier.mergeCategories(cg2, myMeta.masterClassPrediction(m),  myMeta.probs{m});
        probs = probs{1};
        pred = pred{1};
        inds = find( max(probs,[],2) >= confThres(k));
        predThres{m,k} = pred(inds);
    end
        
    statsOutThres{k} = mergeConcentrations(c_titration, cMapInds, predThres(:,k));

    nCells(k) = sum(statsOutThres{k}.nCells);
        
    % Compute each curve's optimal recall and FPR compensation. We weight
    % each term in the curve according to sqrt of the expected number of
    % parasites we should count given the total number of cells we imaged
    % at that point. The reason for doing this is that the highest points
    % have more statistical weight than the the lowest points do. 
    for i = 1:size(F,1)
        for j= 1:size(F,2)
            M(i,j,k) = mean(((c_titration/100).*statsOutThres{k}.nCells').*(((statsOutThres{k}.parasitemiaPercent' - F(i,j))/R(i,j))./c_titration - 1).^2);
        end
    end
    
    bestInd(k) = find(M(:,:,k) == min(min(M(:,:,k))),1);
    recall(k) = R(bestInd(k));
    FPR(k) = F(bestInd(k));

    
    loglog(c_titration, (statsOutThres{k}.parasitemiaPercent)/recall(k) - FPR(k)/recall(k),':o','color',colorSpace(k,:), 'linewidth',1.5); hold all;
%     loglog(c_titration, (statsOutThres{k}.parasitemiaPercent),'o-','color',colorSpace(k,:), 'linewidth',2); hold all;

legStr{end+1} = [num2str(100*confThres(k)), ' %'];
end

grid on;
axis square;
xlim([0.034, 30])
ylim([0.009, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
title({'Leica 405 nm Confidence thresholding','Optimal Recall and FPR compensation'},'fontsize',16);
legend(legStr);

figure('Position',[50,50,300,700]);
subplot(3,1,1)
plot(confThres, recall*100,'k','linewidth',2);
xlim([min(confThres), max(confThres)]);
title('Statistics vs. Threshold');
ylabel('Recall (%)','fontsize',14)
subplot(3,1,2)
plot(confThres, FPR,'k','linewidth',2);
ylabel('False Positive Rate (%)','fontsize',14);
xlim([min(confThres), max(confThres)]);

subplot(3,1,3);
plot(confThres,100* nCells/nCells(1),'k-','linewidth',2);
xlabel('Confidence Threshold (%)','fontsize',16)
ylabel('Cells remaining (%)','fontsize',16);
xlim([min(confThres), max(confThres)]);

%% Human error correction
myMeta.writeSortedInstances(rbcWriteParams, [],[],[], true);

%% Human corrects errors before running this
myMeta.updateAfterHumanAnnotation();

%% Go through datasets and tally results
for m = 1:myMeta.nDatasets
    humanErrCorrPred{m} = myMeta.rbcInstanceImds{m,1,1}.Labels;
end
statsOutErrCorr = mergeConcentrations(c_titration, cMapInds, humanErrCorrPred);

loglog(c_titration, statsOutErrCorr.parasitemiaPercent, 'g*-', 'linewidth',2,'markersize',12); 

%% Make human plots
plotErrorBands(cVec, [100,300,1000,3000,10000,30000,100000]);
hold all;
loglog(c_titration(valCounts(:,5) ~= 0), valCounts(valCounts(:,5) ~= 0,5),'rh--','linewidth',2,'markersize',12);
loglog(c_titration(rebCounts(:,5)~= 0), rebCounts(rebCounts(:,5)~= 0,5),'rp--','linewidth',2,'markersize',12);
loglog(c_titration(madhuraCounts(:,5)~= 0), madhuraCounts(madhuraCounts(:,5)~= 0,5),'rp--','linewidth',2,'markersize',12);
loglog(c_titration, humanMean(:,5), '-sb','linewidth',2,'markersize',12)
legend('1\sigma (N = 100)','1\sigma (N = 300)','1\sigma (N = 1000)','1\sigma (N = 3000)','1\sigma (N = 10000)','1\sigma (N = 30000)', ...
'1\sigma (N = 100000)','Human 1','Human 2','Human 3','Human (average)','location','southeast')
grid on;
axis square;
xlim([0.034, 30])
ylim([0.01, 30])
set(gcf, 'Position',[50,50,750,750]);
set(gca,'fontsize',12);
xlabel('Nominal Parasitemia (%)');
ylabel('Measured Parasitemia (%)');
title({'Manual counting of','Giemsa-stained smears'},'fontsize',16);

%% Error plot

% 405 nm
figure('Position',[50,50,700,300]);
semilogx(c_titration, statsOut.parasitemiaPercent./c_titration', '-o','markersize',6, 'linewidth',1.5);
hold all;
semilogx(c_titration, statsOutErrCorr.parasitemiaPercent./c_titration', '-s','markersize',6, 'linewidth',1.5);

for k=1:numel(confThres)
    semilogx(c_titration, (statsOutThres{k}.parasitemiaPercent/recall(k) - FPR(k)/recall(k))./c_titration', '-o','markersize',6, 'linewidth',1.5,'color',colorSpace(k,:));
end

grid on;
xlim([0.034, 30])
ylim([-0.5, 9]);
set(gca,'YTick',[0:2:9]);
xlabel('Nominal Parasitemia (%)','fontsize',12);
legend('Raw classifier','Human collaboration','Thresholded/Compensated');
