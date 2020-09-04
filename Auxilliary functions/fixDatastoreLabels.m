
function dsOut = fixDatastoreLabels(imdsToFix, imdsCorrect)
% This function repairs labels in the imageDatastore object 'imdsToFix',
% using correct labels in 'imdsCorrect'. 'imdsCorrect' can consist of any
% subset of images from imdsToFix. Labels are corrected if and only if the
% filenames match exactly. Path to the image does not matter. If either of
% the datastores have repeating filenames, we don't deal with that case
% here. 

% Paul Lebel
% czbiohub

dsOut = imdsToFix.copy;

% Extract filenames and labels first as it's much faster than referencing
% the imds objects
[~,origFNames, ~] = cellfun(@fileparts, dsOut.Files, 'UniformOutput', false);
[~,corrFNames, ~] = cellfun(@fileparts, imdsCorrect.Files, 'UniformOutput',false);
corrLabels = imdsCorrect.Labels;

count = 0;
for i=1:numel(corrFNames)
    ind = find(contains(origFNames, corrFNames{i}),1);
    if ~isempty(ind)
        if dsOut.Labels(ind) ~= corrLabels(i)
            dsOut.Labels(ind) = corrLabels(i);            
            count = count+1;
        end
    end
end

disp(['Repaired ', num2str(count), '/',num2str(numel(corrLabels)), ' labels']);