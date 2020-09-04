
% Generate test dataset with appropriate prevalences in each category

classes = {'healthy','ring','troph','schizont'};
prevalences = [0.9125, 0.05, 0.025, 0.0125];
pathToDataset = uigetdir();
testDataset = imageDatastore(pathToDataset,'IncludeSubfolders',true, 'LabelSource','foldernames');
nCellsTotal = 9826;
nCells = nCellsTotal*prevalences;

testDataEq = testDataset.copy;

allInds=  [];
for i=1:4
    classInds = find(testDataset.Labels == classes{i});
    subInds{i} = classInds( randperm(numel(classInds),floor( min(numel(classInds),nCells(i)))));
    allInds = [allInds; subInds{i}];
end

testDataEq.Files = testDataset.Files(allInds);
testDataEq.Labels = testDataset.Labels(allInds);
testDataEq.countEachLabel

[testPred, testProbs] = master4211.classifyDatastore(testDataEq);
master4211.plotConfidenceThresholding(testPred, testDataEq.Labels, testProbs,cg2);


%% Generate slice-by-slice subsets of the data
for i=1:5
    slInds{i} = find(contains(testDataEq.Files, ['sl' num2str(i)]));
    slData{i} = testDataEq.copy;
    slData{i}.Files = testDataEq.Files(slInds{i});
    slData{i}.Labels = testDataEq.Labels(slInds{i});
    [slPred{i}, slProbs{i}] = master4211.classifyDatastore(slData{i});
end

%% Plotting specific slices
% [maxWvStats, maxSlStats, maxOvStats] =  myMeta.maxConfidenceOutput(slPred, slProbs);

for i=1:5
    cm(:,:,i) = confusionmat(slData{i}.Labels, slPred{i})';
    stats(i) = malariaStatsFromCM(cm(:,:,i),true);
    
    master4211.plotConfidenceThresholding(slPred{i}, slData{i}.Labels, slProbs{i});
end
    
% master4211.plotConfidenceThresholding(slPred{j}, slData{j}.Labels, slProbs{j});


