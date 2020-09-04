
% This function serves to transfer human-generated labels from one
% imageDatastore to others after sorting on one channel/slice has been completed. There must be
% one human-sorted imds provided, and any number of machineSorted
% datastores. All the datastores must have images with the same file names
% (but they should be in separate directories), each file corresponding to
% one classified instance.

% Inputs:
% humanSortedImds: One imageDatastore object whose instances have been
% human-sorted.

% machineSortedImds: a cell array of imageDatastore objects with images
% that are parallel to the humanSortedImds and images all named the
% same, but have not been human-annotated. For example, these could be
% from other channels/slices of the same dataset.

% metaData (optional):

% Paul Lebel
% 2019/11/24

function [machineSortedImds, metaData] = updateChannelLabels(humanSortedImds, machineSortedImds, metaData)

if numel(humanSortedImds) > 1
    error('Only one human sorted datastore allowed');
end

if nargin < 2
    error('Not enough input arguments');
end

% Check if machineSortedImds is a cell array. If not, it should be a table
if ~iscell(machineSortedImds)
    if istable(machineSortedImds)
        machineSortedImds = {machineSortedImds};
    else
        error('machineSortedImds must be either a table or a cell array of tables');
    end
end

if nargin < 3
    metaData = cell(size(machineSortedImds));
else
    
% Check if metaData is a cell array. If not, it should be a table
if ~iscell(metaData)
    if istable(metaData)
        metaData = {metaData};
    else
        error('machineSortedImds must be either a table or a cell array of tables');
    end
end

nMachine = numel(machineSortedImds);
nMeta = numel(metaData);
masterLabels = humanSortedImds.Labels;
masterFiles = humanSortedImds.Files;
nFiles = numel(humanSortedImds.Files);

% Use temp cell arrays because accessing imageDatastore files is really
% slow for some reason
machFiles = cell(nMachine,1);
newLabels = machFiles;
metaFiles = cell(nMeta,1);
newTblLabels = metaFiles;

% Initialize new cell arrays of labels
for i=1:nMachine
    machFiles{i} = machineSortedImds{i}.Files;
    newLabels{i} = cell(numel(machFiles{i}),1);
end

for i=1:nMeta
    metaFiles{i} = metaData{i}.Filename;
    newTblLabels{i} = cell(size(metaData{i},1),1);
end

firstMultipleCopyWarning = true;
firstMultipleCopyWarning_tbl = true;

% Transfer the labels from the human-sorted to the machine-sorted databases
for j=1:nFiles
    [~, fName, ~] = fileparts(masterFiles{j});
    
    % In case the set of file names has somehow changed, match the file
    % names when transferring labels
    for i=1:nMachine
        
        imdsInds = find(contains(machFiles{i}, fName));
        
        % Parse cases of number of matching imds file names for each file in the
        % master
        if numel(imdsInds) == 1
            newLabels{i}(imdsInds) = cellstr(masterLabels(j));
        elseif numel(imdsInds) > 1
            if firstMultipleCopyWarning
                warning('More than one matching machine-sorted filename has been found! Updating all matching labels');
                firstMultipleCopyWarning = false;
            end
            for k = 1:numel(imdsInds)
                newLabels{i}(imdsInds(k)) = cellstr(masterLabels(j));
            end
        else
            warning('Filename in machine-sorted imds not found!');
        end
        
    end
    
    for i=1:nMeta
        tblInds = find(contains(metaFiles{i}, fName));
        
        % Parse cases for the filenames in the table
        if numel(tblInds) == 1
            newTblLabels{i}(tblInds) = cellstr(masterLabels(j));
        elseif numel(tblInds) > 1
            if firstMultipleCopyWarning_tbl
                warning('More than one matching machine-sorted filename has been found! Updating all matching labels');
                firstMultipleCopyWarning_tbl = false;
            end
            for k = 1:numel(tblInds)
                newTblLabels{i}(tblInds(k)) = cellstr(masterLabels(j));
            end
        else
            warning('Filename in metaData table not found');
        end
        
    end
    
end

for i=1:nMachine
    machineSortedImds{i}.Labels = categorical(newLabels{i});
end

for i=1:nMeta
        metaData{i}.HumanLabels = categorical(newTblLabels{i});
end


end


