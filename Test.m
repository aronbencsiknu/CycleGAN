clear all; close all; clc;
datasetFolder = fullfile("input","Test","male");
imds = imageDatastore(datasetFolder,IncludeSubfolders=true);
test = augmentedImageDatastore([128,128],imds);

% load trained models
%load("netG_AB_gender_6.mat");
load("netG_AB_gender.mat");

mbqTest = minibatchqueue(test, ...
    MiniBatchSize=1, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");
shuffle(mbqTest);
iteration = 0;
while hasdata(mbqTest)
    iteration = iteration + 1;
    display(iteration);
    testBatch = next(mbqTest);    
    I = extractdata(testBatch(:,:,:,1));
    I = rescale(I);
    
    % save original
    imwrite(I, fullfile("output", append(int2str(iteration), "_original.png")));

    % uncomment for comparison between two models
    %{
    % save modified with 6 residual blocks
    fakeB = predict(netG_AB_gender_6,testBatch(:,:,:,1));
    I = extractdata(fakeB);
    I = rescale(I);
    imwrite(I, fullfile("gender", "Test","compare", append(int2str(iteration), "_modified_6.png")));
    %}

    % save modified with 7 residual blocks
    fakeB = predict(netG_AB_gender,testBatch(:,:,:,1));
    I = extractdata(fakeB);
    I = rescale(I);
    imwrite(I, fullfile("output", append(int2str(iteration), "_modified.png")));
end


function X = preprocessMiniBatch(data)
% Concatenate mini-batch
X = cat(4,data{:});

% Rescale the images in the range [-1 1].
X = rescale(X,-1,1,InputMin=0,InputMax=255);
end

