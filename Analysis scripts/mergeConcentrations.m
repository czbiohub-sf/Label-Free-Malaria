

function statsOut = mergeConcentrations(c_titration, cMapInds, predictionsIn)
    % Sums all the data from distinct datasets acquired for the same titration
    % point. The output is as if each titration point had one unique dataset.

    % Inputs:
    % c_titration is the monotonic vector of concentrations defining the full
    % experiment

    % cMapInds is a vector of indices such that c_titration(cMapInds) is a
    % vector of concentrations directly corresponding to the datasets

    % predictionsIn is a cell array of categorical arrays. It must have the
    % same number of elements as cMapInds (one for each dataset).

    % Vector of concentrations directly corresponding to the datasets
    nCats = numel(categories(predictionsIn{1}));
    summaryArray = zeros(numel(c_titration), nCats);
    
    for m = 1:numel(c_titration)
        inds = find(cMapInds == m);
        for j = inds
            if iscolumn(predictionsIn{j})
                predictionsIn{j} = predictionsIn{j}';
            end
            summaryArray(m,:) = summaryArray(m,:) + countcats(predictionsIn{j});
        end
    end
    
    statsOut.nCells = sum(summaryArray,2);
    statsOut.nParasites = sum(summaryArray(:,2:end),2);
    switch nCats 
        case 2
            statsOut.parasitemiaPercent = statsOut.nParasites*100./statsOut.nCells;
        case 3
            statsOut.nEarly = summaryArray(:,2);
            statsOut.nLate = summaryArray(:,3);
            statsOut.earlyPercent = statsOut.nEarly*100./statsOut.nCells;
            statsOut.latePercent = statsOut.nLate*100./statsOut.nCells;
            statsOut.parasitemiaPercent = statsOut.earlyPercent+statsOut.latePercent;
        case 4
            statsOut.nRings = summaryArray(:,2);
            statsOut.nTrophs = summaryArray(:,4);
            statsOut.nSchizonts = summaryArray(:,3);
            statsOut.ringPercent = statsOut.nRings*100./statsOut.nCells;
            statsOut.trophPercent = statsOut.nTrophs*100./statsOut.nCells;
            statsOut.schizontPercent = statsOut.nSchizonts*100./statsOut.nCells;
            statsOut.parasitemiaPercent = statsOut.ringPercent+statsOut.trophPercent+statsOut.schizontPercent;
    end
    
    
    
    % Compute the counting statistics -- Poisson error is sqrt of the total
    % number of counts. 
    statsOut.parasitemiaPercentError = 100*sqrt(statsOut.nParasites)./statsOut.nCells;
end

