function img = convertTo8bit(filename)

% Average over color channels if present
img = mean(double(imread(filename)),3);

% Use histogram-based rescaling to be robust against hot pixels
[a,xout] = hist(img(:),10000);
csum = cumsum(a);
csum = csum/max(csum);
maxInd = find(csum > .999,1);
ma = xout(maxInd);
minInd = find(csum > .001,1);
mi = xout(minInd);

img = uint8(255*(double(img)-mi)/(ma-mi));
end