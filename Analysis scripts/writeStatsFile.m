function writeStatsFile(pathname, filename, statsIn, prefix, fid, isTop)

% Recursive function that traverses a nested tree struct, writing a text file
% line by line, consisting of the numerical values at the bottom of the
% tree. Each line contains a prefix, as well as the text formatting
% requirements for a latex .sty file. 

% The purpose of this function is to automatically create a .sty file
% consisting of numerical statistics formatted as LaTex commands. 

if nargin < 6
    isTop = true;
end

if nargin < 5
    fid = fopen(fullfile(pathname, filename), 'a');
end

if nargin < 4
    prefix = '\\newcommand\\';
end

if isTop
    fprintf(fid, '\n\n');
end

fNames = fields(statsIn);

for i = 1:numel(fNames)
    if isstruct(statsIn.(fNames{i}))
        
        % Recursive call to get to the bottom of the tree
        writeStatsFile(pathname, filename, statsIn.(fNames{i}), ...
            [prefix, fNames{i}], fid, false);
    else
        fprintf(fid,[prefix, fNames{i}, '{', num2str(statsIn.(fNames{i}), '%3.2f'), '}\n']);
    end
end

if isTop
    fclose(fid);
end

end