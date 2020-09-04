% Demo script making use of an MDImage object for some basic pre-processing
% operations. 
% Paul Lebel
% czbiohub

myImage = MDImage([1040 1392]/2);
pathname = 'C:\Documents\RawData\';
prefix = 'Malaria';
outputPath = 'PathToDesiredOutput';

myImage.setFilePath(pathname, prefix);
%%
myImage.importMetadata();
myImage.setBGImage('x265nm_BF',bgImage_00);
myImage.setBGImage('x285nm_BF',bgImage_01);
%%
myImage.displayImages();
%%
myImage.refocusImages('gradient',2);
myImage.exportCorrectedImages(outputPath,'AVGrefocusedR2',true);
myImage.displayImages();
%%
myImage.unloadCorrectedImages(true);
myImage.transportOfIntensity('',true,3);
myImage.exportCorrectedImages(outputPath,'TIE',true);