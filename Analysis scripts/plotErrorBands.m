
function plotErrorBands(parasitemiaPercent, nCellsContours)
% This function is highly specific to creating shaded overlays on a log-log
% plot, consisting of regions on that 2D plane which span +/- one standard
% deviation of a Poisson distribution. Many such shaded regions can be
% overlaid, each corresponding to a different parameter set (total number
% of cells counted). 

% parasitemiaPercent is a vector of x-values of known titration points (in
% percent). A plot will be made with this vector as the x-values, and
% error bands computed at these points.

% nCellsContours is a vector defining the parameters of the contours over
% which to plot error bands
figure();
axis square;
grid;
hold all;

if ~isrow(parasitemiaPercent)
    parasitemiaPercent = parasitemiaPercent';
end

parasitemiaPercent = sort(parasitemiaPercent, 'ascend');

% Sort these so that the smallest counts (biggest error bands) get plotted
% first)
nCellsContours = sort(nCellsContours, 'ascend');

cMap = flipud(gray());

for i = 1:numel(nCellsContours)
        
    % Plot a fill area for the stdev errorbars
    errorsTop = parasitemiaPercent + 10*sqrt(parasitemiaPercent)/sqrt(nCellsContours(i));
    errorsBottom = parasitemiaPercent - 10*sqrt(parasitemiaPercent)/sqrt(nCellsContours(i));
    errorsBottom = max(errorsBottom, 0.001);
   
    fill([parasitemiaPercent, fliplr(parasitemiaPercent)],...
        [errorsTop, fliplr(errorsBottom)],cMap(ceil(255*i/numel(nCellsContours)), :), ...
        'linestyle','none');
end
set(gca, 'xscale','log');
set(gca, 'yscale','log');