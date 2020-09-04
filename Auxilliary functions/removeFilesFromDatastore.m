
function imdsOut = removeFilesFromDatastore(imds, filenames)

% Removes images from an imageDatastore object. 
% filenames is a cell array of strings containing the image filename to be
% removed. filenames should not contain the file extensions. 

imdsOut = imds.copy;

files = imds.Files;
labels = imds.Labels;

% Find the indice(s) corresponding to the given file name
[~,fNames,~] = cellfun(@fileparts, files, 'UniformOutput',false);

inds = false(numel(files),1);
for i=1:numel(filenames)
    inds = inds|contains(fNames, filenames{i});
end

% Invert inds and assign to new datastore
inds = ~inds;
imdsOut.Files = files(inds);
imdsOut.Labels = labels(inds);