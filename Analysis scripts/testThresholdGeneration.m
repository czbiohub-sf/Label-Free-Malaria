
% Explore two different methods for producing confidence thresholds as well
% as human collaboration

% Human collaboration. Here we simply set a low-confidence fraction to the
% correct labels and re-analyze
numberToCorrect = 1:100:5000;
YPred = myMeta.masterClassifier.YPred;
probs = myMeta.masterClassifier.probs;
nTotal = numel(YPred);

% [YPred, probs] = myMeta.masterClassifier.classifyDatastore();
gTruth = myMeta.masterClassifier.imdsVal.Labels;
% plotconfusion(gTruth, YPred,'Original');
cmOrig = confusionmat(gTruth, YPred)';
[sortedMaxProbs, IX] = sort(max(probs,[],2),'ascend');

for i=1:numel(numberToCorrect)
    YPredCorrected = YPred;
    YPredCorrected(IX(1:numberToCorrect(i))) = gTruth(IX(1:numberToCorrect(i)));
    cmCorrected = confusionmat(gTruth, YPredCorrected)';
    stats(i) = malariaStatsFromCM(cmCorrected,true);
end

figure;
subplot(2,2,1);
plot(100*numberToCorrect/nTotal, [stats.AllOvAcc],'linewidth',2);
ylabel('Overall Accuracy (%)','fontsize',14);
xlabel('Percent corrected (%)','fontsize',14);
title('Human collaboration','fontsize',14);

subplot(2,2,2);
plot(100*numberToCorrect/nTotal, [stats.RingFpr],'linewidth',2); hold all;
plot(100*numberToCorrect/nTotal, [stats.TrophFpr],'linewidth',2);
plot(100*numberToCorrect/nTotal, [stats.SchizontFpr],'linewidth',2);
ylabel('False-positive rates (%)','fontsize',14);
xlabel('Percent corrected (%)','fontsize',14);

subplot(2,2,3);
plot(100*numberToCorrect/nTotal, [stats.HealthyPrec],'linewidth',2); hold all;
plot(100*numberToCorrect/nTotal, [stats.RingPrec],'linewidth',2); hold all;
plot(100*numberToCorrect/nTotal, [stats.TrophPrec],'linewidth',2);
plot(100*numberToCorrect/nTotal, [stats.SchizontPrec],'linewidth',2);
ylabel('Precision (%)','fontsize',14);
xlabel('Percent corrected (%)','fontsize',14);
legend('Healthy','Rings','Trophs','Schizonts');

subplot(2,2,4);
plot(100*numberToCorrect/nTotal, [stats.HealthyRecall],'linewidth',2); hold all;
plot(100*numberToCorrect/nTotal, [stats.RingRecall],'linewidth',2); hold all;
plot(100*numberToCorrect/nTotal, [stats.TrophRecall],'linewidth',2);
plot(100*numberToCorrect/nTotal, [stats.SchizontRecall],'linewidth',2);
ylabel('Recall (%)','fontsize',14);
xlabel('Percent corrected (%)','fontsize',14);

% FPR Parameterization
healthyDataInd = 3;
[healthyPred, healthyProbs] = myMeta.masterClassifier.classifyDatastore(myMeta.rbcInstanceImds{healthyDataInd, 2});
threshold = 50:99;
myMeta.masterClassifier.plotConfidenceThresholding(healthyPred,...
    myMeta.rbcInstanceImds{healthyDataInd,2}.Labels,...
    healthyProbs)    

% Population sub-sampling