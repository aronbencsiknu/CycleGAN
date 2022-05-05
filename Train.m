resize = [128,128];
%load domain A
datasetFolder = fullfile("input","Train","male");
imds = imageDatastore(datasetFolder,IncludeSubfolders=true);
augmenter = imageDataAugmenter(RandXReflection=true);
augimdsA = augmentedImageDatastore(resize,imds,DataAugmentation=augmenter);

% load domain B
datasetFolder = fullfile("input","Train","female");
imds = imageDatastore(datasetFolder,IncludeSubfolders=true);
augmenter = imageDataAugmenter(RandXReflection=true);
augimdsB = augmentedImageDatastore(resize,imds,DataAugmentation=augmenter);

% load domain fordisplay (identical to domain A)
datasetFolder = fullfile("input","Train","male");
imds = imageDatastore(datasetFolder,IncludeSubfolders=true);
test = augmentedImageDatastore([128,128],imds);

% initialization variables
settings.lambda_cycle = 10;
settings.lambda_id = .1*settings.lambda_cycle;
numEpochs = 250;
miniBatchSize = 32;
learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
validationFrequency = 50;

% define training batches
augimdsA.MiniBatchSize = miniBatchSize;
mbqA = minibatchqueue(augimdsA, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

augimdsB.MiniBatchSize = miniBatchSize;
mbqB = minibatchqueue(augimdsB, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

% define test batch
augimdsB.MiniBatchSize = miniBatchSize;
mbqTest = minibatchqueue(test, ...
    MiniBatchSize=miniBatchSize, ...
    PartialMiniBatch="discard", ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB");

% initialize generators
imageSize = [resize,3];
netG_BA = generator(imageSize);
netG_AB = generator(imageSize);
netGTest_AB = generator(imageSize);

% initialize discriminators
netD_A = discriminator(imageSize);
netD_B = discriminator(imageSize);

% average Gradient and average Gradient squared holders
trailingAvgG_AB = [];
trailingAvgSqG_AB = [];
trailingAvgG_BA = [];
trailingAvgSqG_BA = [];

trailingAvgD_A = [];
trailingAvgSqD_A = [];
trailingAvgD_B = [];
trailingAvgSqD_B = [];

% variables for plotting
f = figure;
imageAxes = subplot(2,2,1);
imageAxes1 = subplot(2,2,2);
scoreAxes = subplot(2,2,3);

lineWidth = 1.5;
C = colororder;
lineScoreG = animatedline(scoreAxes,Color=C(1,:),LineWidth=lineWidth);
legend("Generator X + Generator Y");
ylim([0 20])
xlabel("Iteration")
ylabel("Loss")
grid on
scoreAxes1 = subplot(2,2,4);
C = colororder;
lineScoreD_A = animatedline(scoreAxes1,Color=C(2,:),LineWidth=lineWidth);
lineScoreD_B = animatedline(scoreAxes1,Color=C(3,:), LineWidth=lineWidth);
legend("Discriminator X", "Discriminator Y");
ylim([0 1])
xlabel("Iteration")
ylabel("Loss")
grid on

iteration = 0;
start = tic;
%% Train
% Loop over epochs.
for epoch = 1:numEpochs

    % Reset and shuffle datastore.
    shuffle(mbqA);
    shuffle(mbqB);
    shuffle(mbqTest);

    % Loop over mini-batches.
    while hasdata(mbqA)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        ABatch= next(mbqA);
        BBatch= next(mbqB);

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [G_loss,D_A_loss,D_B_Loss,gradientsG_BA, gradientsG_AB, gradientsD_A, gradientsD_B, stateG, stateG1] = ...
            dlfeval(@modelLoss,netG_BA,netG_AB, netD_A, netD_B, ABatch,BBatch, settings);
        netG_BA.State = stateG;
        netG_AB.State = stateG1;
        % Update the discriminator A network parameters.
        [netD_A,trailingAvgD_A,trailingAvgSqD_A] = adamupdate(netD_A, gradientsD_A, ...
            trailingAvgD_A, trailingAvgSqD_A, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the discriminator B network parameters.
        [netD_B,trailingAvgD_B,trailingAvgSqD_B] = adamupdate(netD_B, gradientsD_B, ...
            trailingAvgD_B, trailingAvgSqD_B, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the generator BA network parameters.
        [netG_BA,trailingAvgG_BA,trailingAvgSqG_BA] = adamupdate(netG_BA, gradientsG_BA, ...
            trailingAvgG_BA, trailingAvgSqG_BA, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the generator AB network parameters.
        [netG_AB,trailingAvgG_AB,trailingAvgSqG_AB] = adamupdate(netG_AB, gradientsG_AB, ...
            trailingAvgG_AB, trailingAvgSqG_AB, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Every validationFrequency iterations, display batch of generated
        % images using the held-out generator input.
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            testBatch = next(mbqTest);
            
            % Tile and rescale the images in the range [0 1].
            I = extractdata(testBatch(:,:,:,1));
            I = rescale(I);

            % Display the images.
            subplot(2,2,1);
            image(imageAxes,I)
            
            xticklabels([]);
            yticklabels([]);
            title("Original");
            
            fakeB = predict(netG_AB,testBatch(:,:,:,1));
            
            % Tile and rescale the images in the range [0 1].
            I = extractdata(fakeB);
            I = rescale(I);

            % Display the images.
            subplot(2,2,2);
            image(imageAxes1,I)
            xticklabels([]);
            yticklabels([]);
            title("Manipulated");
        end
        if epoch > 240
            netG_BA_gender = netG_BA;
            netG_AB_gender = netG_AB;
            netD_A_gender = netD_A;
            netD_B_gender = netD_B;

            % save trained networks
            save("netG_BA_gender.mat", "netG_BA_gender");
            save("netG_AB_gender.mat", "netG_AB_gender");
            save("netD_A_gender.mat", "netD_A_gender");
            save("netD_B_gender.mat", "netD_B_gender");
        end
        % Update the scores plot.
        subplot(2,2,3)
        addpoints(lineScoreG,iteration,double(gatext(G_loss)));
        % Update the title with training progress information.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        title("Elapsed: " + string(D))
        drawnow

        subplot(2,2,4)
        addpoints(lineScoreD_A,iteration,double(gatext(D_A_loss)));
        addpoints(lineScoreD_B,iteration,double(gatext(D_B_Loss)));
        % Update the title with training progress information.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration)
        drawnow
    end
end
%% Loss function
function [g_loss,dA_loss, dB_loss,gradientsG, gradientsG1,gradientsD, gradientsD1,stateG, stateG1] = ...
    modelLoss(netG_BA, netG_AB, netD_A, netD_B, A,B, settings)

% Calculate the results for not in-domain data with the generator networks.
[fakeA, stateG] = forward(netG_BA,B);
[fakeB, stateG1] = forward(netG_AB,A);

% Calculate the results reconstruction with the generator networks.
reconB = forward(netG_AB, fakeA);
reconA = forward(netG_BA, fakeB);

% Calculate the results for already in-domain data with the generator networks.
idA = forward(netG_BA, A);
idB = forward(netG_AB, B);

% Calculate the predictions for real data with the discriminator networks.
validD_A= forward(netD_A, A);
validD_B= forward(netD_B, B);

% Calculate the predictions for fake data with the discriminator networks.
fakeD_A= forward(netD_A, fakeA);
fakeD_B= forward(netD_B, fakeB);

% Calculate the GAN loss.

% Adverserial loss
dA_loss_real = mean((validD_A-1).^2,'all');
dA_loss_fake = mean((fakeD_A).^2,'all');
dA_loss = .5*(dA_loss_real+dA_loss_fake);

dB_loss_real = mean((validD_B-1).^2,'all');
dB_loss_fake = mean((fakeD_B).^2,'all');
dB_loss = .5*(dB_loss_real+dB_loss_fake);

gAB_loss_fake = mean((fakeD_B-1).^2,'all');
gBA_loss_fake = mean((fakeD_A-1).^2,'all');

% Cycle-consistency loss
gAB_L1re = mean(abs(reconB-B),'all');
gBA_L1re = mean(abs(reconA-A),'all');

% Identity loss
gAB_L1id = mean(abs(idB-B),'all');
gBA_L1id = mean(abs(idA-A),'all');

id_total = gAB_L1id + gBA_L1id;

%Total generator loss
g_loss=gAB_loss_fake+gBA_loss_fake+...
    settings.lambda_cycle*(gAB_L1re+gBA_L1re)+...
    settings.lambda_id*(id_total);

% For each network, calculate the gradients with respect to the loss.
[gradientsG, gradientsG1] = dlgradient(g_loss, netG_BA.Learnables, netG_AB.Learnables,RetainData=true);
gradientsD = dlgradient(dA_loss,netD_A.Learnables);
gradientsD1 = dlgradient(dB_loss,netD_B.Learnables);
end
%% Generator definition
function lgraph = generator(inputSize)

lgraph = layerGraph();
% downsample
tempLayers = [
    imageInputLayer(inputSize,"Name","imageinput","Normalization","none")
    convolution2dLayer([4 4],64,"Name","conv_1","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([4 4],128,"Name","conv_2","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    convolution2dLayer([4 4],256,"Name","conv_3","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);
% 7 residual blocks
tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_4","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_5","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_6","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_7","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_8","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_9","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","conv_10","Padding",[1 1 1 1])
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);
% upsample
tempLayers = [
    additionLayer(2,"Name","addition_7")
    transposedConv2dLayer([4 4],128,"Name","transposed-conv_1","Cropping","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_11")
    transposedConv2dLayer([4 4],128,"Name","transposed-conv_2","Cropping","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_12")
    transposedConv2dLayer([4 4],inputSize(3),"Name","transposed-conv_3","Cropping","same","Stride",[2 2])
    tanhLayer("Name","tanh")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% connect layers
lgraph = connectLayers(lgraph,"relu_3","conv_4");
lgraph = connectLayers(lgraph,"relu_3","addition_1/in1");
lgraph = connectLayers(lgraph,"relu_4","addition_1/in2");
lgraph = connectLayers(lgraph,"addition_1","conv_5");
lgraph = connectLayers(lgraph,"addition_1","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_5","addition_2/in2");
lgraph = connectLayers(lgraph,"addition_2","conv_6");
lgraph = connectLayers(lgraph,"addition_2","addition_3/in2");
lgraph = connectLayers(lgraph,"relu_6","addition_3/in1");
lgraph = connectLayers(lgraph,"addition_3","conv_7");
lgraph = connectLayers(lgraph,"addition_3","addition_4/in2");
lgraph = connectLayers(lgraph,"relu_7","addition_4/in1");
lgraph = connectLayers(lgraph,"addition_4","conv_8");
lgraph = connectLayers(lgraph,"addition_4","addition_5/in1");
lgraph = connectLayers(lgraph,"relu_8","addition_5/in2");
lgraph = connectLayers(lgraph,"addition_5","conv_9");
lgraph = connectLayers(lgraph,"addition_5","addition_6/in1");
lgraph = connectLayers(lgraph,"relu_9","addition_6/in2");
lgraph = connectLayers(lgraph,"addition_6","conv_10");
lgraph = connectLayers(lgraph,"addition_6","addition_7/in2");
lgraph = connectLayers(lgraph,"relu_10","addition_7/in1");
lgraph = dlnetwork(lgraph);

end

%% Discriminator definition
function lgraph = discriminator(inputSize)
% downsample
layers = [
    imageInputLayer(inputSize,"Name","imageinput","Normalization","none")
    convolution2dLayer([4 4],64,"Name","conv_1","Padding",[1 1 1 1],"Stride",[2 2])
    reluLayer("Name","relu_1")
    convolution2dLayer([4 4],128,"Name","conv_2","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")
    convolution2dLayer([4 4],256,"Name","conv_3","Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_3")
    convolution2dLayer([1 1],1,"Name","conv_4",'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'))];

    lgraph = layerGraph(layers);
    lgraph = dlnetwork(lgraph);
end

function X = preprocessMiniBatch(data)
% Concatenate mini-batch
X = cat(4,data{:});

% Rescale the images in the range [-1 1].
X = rescale(X,-1,1,InputMin=0,InputMax=255);
end

%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end