
% This function's role is to take a struct of data statistics as input and
% write them to a file in a specific format that allows them to be read in
% as commands in overleaf (online LaTeX editor). The purpose is to have
% these coded into the manuscript as variables that can change after
% analysis is updated, without having to generate and copy them over by
% hand.

% The function is recursive, allowing a struct with a tree structure to be
% traversed with minimal code.

% Manuscript: https://www.overleaf.com/project/5e34a8c7755aa60001928849

% Where "newcommand\" is a constant, literal string that is by default written
% as a prefix for LaTeX to recognize it as a command that can be imported.

% Command formatting is done in the following way because we can't have
% numbers or any other non-alphabetical characters:
% Scope-Channel-Thres-Cons-Cats-Stage-Type

% (No hyphens used in command)

% Where:
% Scope = scope used (uv / leica)
% Channel =  for uv scope: Cha = 285 nm, Chb = 365 nm, Chc = 565 nm
%           for leica: Cha = 365 nm, Chb = 405 nm, Chc = lamp

% Thres = Whether confidence thresholding was used or not. "NoThres" or
% "Thres"

% Cons = Whether consensus (max confidence) was used or not. Either
% "NoConst", "SlCons","WvCons", or "SlWvCons".

% Cats = Granularity of the measurement. "fg" means no coarse-graining,
% "cg" means coarse-grained (3 categories), and "bin" means binary (2
% categories).

% Stage = Life cycle stages. This is a heterogeneous branch as the stages
% are different for the different levels of granularity.

% Type: Type of statistic being reported. Exs:
%       OvAcc = Overall Accuracy
%       FPR = False Positive Rate
%       Recall
%       Prec = Precision
%       CErr = Sample Composition Error

pathname = 'C:\Users\SingleCellPicker\Documents';
filename = 'testStatsFile.sty';

stages_fg = {'All','Healthy','Ring','Troph','Schizont'};
stages_cg = {'All','Healthy','Early','Late'};
stages_bin = {'All','Healthy','Para'};

fieldsIn = {...
    {'Uv','Leica'};...
    {'Cha','Chb','Chc'};...
    {'NoThres','Thres'};...
    {'NoCons','SlCons','WvCons','SlWvCons'};...
    {'Fg','Cg','Bin'};...
    {stages_fg, stages_cg, stages_bin};...
    {'OvAcc','Fpr','Recall','Prec','Cerr'}};

stats = buildNestedStruct(fieldsIn, -1);

writeStatsFile(pathname, filename, stats);