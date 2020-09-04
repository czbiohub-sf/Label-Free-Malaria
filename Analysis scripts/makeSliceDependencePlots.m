%% Create slice consensus plots
% Load single wavelength meta object

fg.OldCats = {'healthy','ring','troph','schizont'};
fg.NewCats = fg.OldCats;
cg.OldCats = {{'troph', 'schizont'}};
cg.NewCats = {'late'};
cg2.OldCats = {{'ring','troph','schizont'}};
cg2.NewCats = {'parasitized'};

% Run the master classifier on the separate slices
mcSliceSepPred = cell(1,myMeta.nSlices);
mcSliceSepProbs = cell(1,myMeta.nSlices, 1);
cmss = zeros(myMeta.nClasses, myMeta.nClasses, myMeta.nSlices);
cmssNormPercent = zeros(myMeta.nClasses, myMeta.nClasses, myMeta.nSlices);
mcOvAccVsSlice = zeros(myMeta.nSlices,1);
sliceDepGroundTruth = myMeta.rbcInstancesSlicesSeparate{myMeta.pipeOpts.masterChannelInd,1}.Labels;

for j=1:myMeta.nSlices
    [mcSliceSepPred{1,j}, mcSliceSepProbs{1,j}] = ...
        myMeta.masterClassifier.classifyDatastore(...
        myMeta.rbcInstancesSlicesSeparate{myMeta.pipeOpts.masterChannelInd,j});
    
    % Use transpose of confusionmat output because its format is not
    % consistent with matlab's 'plotconfusion'
    cmss(:,:,j) = confusionmat(sliceDepGroundTruth, mcSliceSepPred{1,j})';
    cmssNormPercent(:,:,j) = 100*cmss(:,:,j)/sum(sum(cmss(:,:,j)));
    mcOvAccVsSlice(j) = 100*trace(cmss(:,:,j))/sum(sum(cmss(:,:,j)));    
end

trueCounts = countcats(myMeta.rbcInstancesSlicesSeparate{myMeta.pipeOpts.masterChannelInd,j}.Labels);
trueCountsPercent = 100*trueCounts/sum(trueCounts);

%% Make plots for slice-dependent CM statistics
count = 0;
deltaZVec = -1:.5:1;

for i=1:myMeta.nClasses
    for j=1:myMeta.nClasses
        count = count+1;
        subplot(myMeta.nClasses,myMeta.nClasses,sub2ind([myMeta.nClasses, myMeta.nClasses], j,i));
        plot(deltaZVec,squeeze(cmss(i,j,:)),'linewidth',2);
        title({['Human: ', myMeta.classes{j}], [' Predicted: ', myMeta.classes{i}]});
        if (i == myMeta.nClasses) && (j == 2)
            xlabel('Focal shift (\mum)','fontsize',14);
        end
        drawnow;
    end
end

% Perform slice consensus
[~, mcMaxSlStats, ~] = myMeta.maxConfidenceOutput(mcSliceSepPred, mcSliceSepProbs);
cmSliceCons = confusionmat(myMeta.rbcInstancesSlicesSeparate{1,1}.Labels, mcMaxSlStats.pred')';
figure;
plotconfusion(myMeta.rbcInstancesSlicesSeparate{1,1}.Labels,mcMaxSlStats.pred');
cmSliceConsNormPercent = 100*cmSliceCons/(sum(sum(cmSliceCons)));


%% Plotting stats vs. slice number and consensus (max confidence)
% Compute false positive rates
figure('Position',[300 0 630 750]);

mcRingFPRvsSlice = squeeze(cmssNormPercent(2,1,:));
mcTrophFPRvsSlice = squeeze(cmssNormPercent(4,1,:));
mcSchizFPRvsSlice = squeeze(cmssNormPercent(3,1,:));
mcRingFPR_sliceCons = cmSliceConsNormPercent(2,1);
mcTrophFPR_sliceCons = cmSliceConsNormPercent(4,1);
mcSchizFPR_sliceCons = cmSliceConsNormPercent(3,1);


subplot(3,2,1); set(gca, 'Fontsize',12);
plot(deltaZVec, 100*mcRingFPRvsSlice, 'bo-', 'linewidth',2); hold all;
plot(deltaZVec, 100*mcTrophFPRvsSlice, 'ro-', 'linewidth',2);
plot(deltaZVec, 100*mcSchizFPRvsSlice, 'yo-', 'linewidth',2);
plot(deltaZVec, 100*cmSliceConsNormPercent(2,1)*ones(numel(deltaZVec),1),'b-.','markersize',12);
plot(deltaZVec, 100*cmSliceConsNormPercent(4,1)*ones(numel(deltaZVec),1),'r-.','markersize',12);
plot(deltaZVec, 100*cmSliceConsNormPercent(3,1)*ones(numel(deltaZVec),1),'y-.','markersize',12);
set(gca,'XTick',deltaZVec);
title('False positive rate vs. slice');
xlabel('Defocus from global best (\mum)','fontsize',12);
ylabel({'False positives','per 10,000 cells'},'fontsize',12);
set(gca, 'Fontsize',12);

% Compute recall vs. slice
mcHealthyRecallVsSlice = 100*squeeze(cmss(1,1,:))./squeeze(sum(cmss(:,1,:),1));
mcRingRecallVsSlice = 100*squeeze(cmss(2,2,:))./squeeze(sum(cmss(:,2,:),1));
mcTrophRecallVsSlice = 100*squeeze(cmss(4,4,:))./squeeze(sum(cmss(:,4,:),1));
mcSchizontRecallVsSlice = 100*squeeze(cmss(3,3,:))./squeeze(sum(cmss(:,3,:),1));
sliceConsRecallHealthy = 100*cmSliceConsNormPercent(1,1)/sum(cmSliceConsNormPercent(:,1),1);
sliceConsRecallRing = 100*cmSliceConsNormPercent(2,2)/sum(cmSliceConsNormPercent(:,2),1);
sliceConsRecallSchizont = 100*cmSliceConsNormPercent(3,3)/sum(cmSliceConsNormPercent(:,3),1);
sliceConsRecallTroph = 100*cmSliceConsNormPercent(4,4)/sum(cmSliceConsNormPercent(:,4),1);

subplot(3,2,2);
plot(deltaZVec, mcHealthyRecallVsSlice, 'ko-', 'linewidth',2); hold all;
plot(deltaZVec, mcRingRecallVsSlice, 'bo-', 'linewidth',2);
plot(deltaZVec, mcTrophRecallVsSlice, 'ro-', 'linewidth',2);
plot(deltaZVec, mcSchizontRecallVsSlice, 'yo-', 'linewidth',2);
plot(deltaZVec, sliceConsRecallHealthy*ones(numel(deltaZVec),1),'k-.','markersize',12);
plot(deltaZVec, sliceConsRecallRing*ones(numel(deltaZVec),1),'b-.','markersize',12);
plot(deltaZVec, sliceConsRecallTroph*ones(numel(deltaZVec),1),'r-.','markersize',12);
plot(deltaZVec, sliceConsRecallSchizont*ones(numel(deltaZVec),1),'y-.','markersize',12);
set(gca,'XTick',deltaZVec);
title('Recall vs. slice');
xlabel('Defocus from global best (\mum)','fontsize',12);
ylabel('Recall (%)','fontsize',12);
legend('Healthy','Ring', 'Troph', 'Schizont', 'Healthy - max confidence', 'Ring - max confidence slice', 'Troph - max confidence slice','Schizont - max confidence slice');
set(gca, 'Fontsize',12);

% Compute precision vs. slice
mcHealthyPrecVsSlice = 100*squeeze(cmss(1,1,:))./squeeze(sum(cmss(1,:,:),2));
mcRingPrecVsSlice = 100*squeeze(cmss(2,2,:))./squeeze(sum(cmss(2,:,:),2));
mcTrophPrecVsSlice = 100*squeeze(cmss(4,4,:))./squeeze(sum(cmss(4,:,:),2));
mcSchizontPrecVsSlice = 100*squeeze(cmss(3,3,:))./squeeze(sum(cmss(3,:,:),2));

sliceConsPrecHealthy = 100*cmSliceConsNormPercent(1,1)/sum(cmSliceConsNormPercent(1,:),2);
sliceConsPrecRing = 100*cmSliceConsNormPercent(2,2)/sum(cmSliceConsNormPercent(2,:),2);
sliceConsPrecSchizont = 100*cmSliceConsNormPercent(3,3)/sum(cmSliceConsNormPercent(3,:),2);
sliceConsPrecTroph = 100*cmSliceConsNormPercent(4,4)/sum(cmSliceConsNormPercent(4,:),2);

subplot(3,2,3);
plot(deltaZVec, mcHealthyPrecVsSlice, 'ko-', 'linewidth',2); hold all;
plot(deltaZVec, mcRingPrecVsSlice, 'bo-', 'linewidth',2);
plot(deltaZVec, mcTrophPrecVsSlice, 'ro-', 'linewidth',2);
plot(deltaZVec, mcSchizontPrecVsSlice, 'yo-', 'linewidth',2);
plot(deltaZVec, sliceConsPrecHealthy*ones(numel(deltaZVec),1),'k-.');
plot(deltaZVec, sliceConsPrecRing*ones(numel(deltaZVec),1),'b-.');
plot(deltaZVec, sliceConsPrecTroph*ones(numel(deltaZVec),1),'r-.');
plot(deltaZVec, sliceConsPrecSchizont*ones(numel(deltaZVec),1),'y-.');
set(gca,'XTick',deltaZVec);
title('Precision vs. slice');
xlabel('Defocus from global best (\mum)','fontsize',12);
ylabel('Precision (%)','fontsize',12);
set(gca, 'Fontsize',12);

sliceConsOvAcc = 100*trace(cmSliceConsNormPercent)/sum(sum(cmSliceConsNormPercent));
subplot(3,2,4); 
plot(deltaZVec, mcOvAccVsSlice, 'ko-', 'linewidth',2); 
hold all;
plot(deltaZVec,  sliceConsOvAcc*ones(numel(deltaZVec),1), 'k-.');
xlabel('Defocus from global best (\mum)','fontsize',12);
ylabel('Overall Accuracy (%)');
set(gca,'XTick',deltaZVec);
title('Overall Accuracy vs. Slice');
ylim([98,100]);
set(gca, 'Fontsize',12);

% Overall counts plot
sliceConsRingCount = sum(cmSliceConsNormPercent(2,:));
sliceConsTrophCount = sum(cmSliceConsNormPercent(3,:));
sliceConsSchizontCount = sum(cmSliceConsNormPercent(4,:));

% True counts
subplot(3,2,5);
plot(deltaZVec, squeeze(sum(cmssNormPercent(2,:,:),2))-trueCountsPercent(2), 'bo-', 'linewidth',2);
hold all;
plot(deltaZVec, squeeze(sum(cmssNormPercent(3,:,:),2))-trueCountsPercent(3), 'ro-', 'linewidth',2);
plot(deltaZVec, squeeze(sum(cmssNormPercent(4,:,:),2))-trueCountsPercent(4), 'yo-', 'linewidth',2);
plot(deltaZVec, sliceConsRingCount*ones(numel(deltaZVec),1)-trueCountsPercent(2),'b-.');
plot(deltaZVec, sliceConsTrophCount*ones(numel(deltaZVec),1)-trueCountsPercent(3),'r-.');
plot(deltaZVec, sliceConsSchizontCount*ones(numel(deltaZVec),1)-trueCountsPercent(4),'y-.');

xlabel('Defocus from global best (\mum)','fontsize',12);
ylabel('Composition error (%)','fontsize',12);
set(gca, 'Fontsize',12);
%%
% Confidence thresholding slice-consencus results
[uvnm285ThresSlConsGT, uvnm285ThresSlConsPred] = myMeta.masterClassifier.plotConfidenceThresholding(mcMaxSlStats.pred', myMeta.rbcInstancesSlicesSeparate{1,1}.Labels, squeeze(mcMaxSlStats.probs));
stats.uv.nm285.Thres.SlCons = myMeta.masterClassifier.computeStats(uvnm285ThresSlConsPred, uvnm285ThresSlConsGT);

% Confidence thresholding and coarse-grained 1
[~,~,~, uvnm285CgThresSlConsPred] = myMeta.masterClassifier.plotConfidenceThresholding(mcMaxSlStats.pred', myMeta.rbcInstancesSlicesSeparate{1,1}.Labels, squeeze(mcMaxSlStats.probs), cg);

[~,~,~, uvnm285BinThresSlConsPred] = myMeta.masterClassifier.plotConfidenceThresholding(mcMaxSlStats.pred', myMeta.rbcInstancesSlicesSeparate{1,1}.Labels, squeeze(mcMaxSlStats.probs), cg2);

