

% This function simply creates a new image datastore object that is the
% combination of multiple individual datastores
function imdsCombined = combineIMDS(imdsCellArray)

nDatastores = numel(imdsCellArray);
fileList = {};
labels = {};
for i = 1:nDatastores
    fileList = cat(1,fileList, imdsCellArray{i}.Files);
    labels = cat(1,labels, imdsCellArray{i}.Labels);
end

imdsCombined = imageDatastore(fileList);
imdsCombined.Labels = labels;