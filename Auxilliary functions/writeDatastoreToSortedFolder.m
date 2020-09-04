
function writeDatastoreToSortedFolder(imds, outputDir)

% Writes the contents of an imageDatastore object to labeled folders. 

classes = categories(imds.Labels);
nFiles = numel(imds.Files);
[~,fNames, exts] = cellfun(@fileparts, imds.Files, 'UniformOutput', false);

% Create labeled folders
for i=1:numel(classes)
    classFolder = fullfile(outputDir, classes{i});
    if ~exist(classFolder,'dir')
        mkdir(classFolder);
    end
end

for i=1:nFiles
    destFile = fullfile(outputDir,string(imds.Labels(i)), [fNames{i}, exts{i}]);
    copyfile(imds.Files{i}, destFile);
end