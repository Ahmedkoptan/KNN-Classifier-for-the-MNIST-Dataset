clear all
close all
clc


%accessing all the necessary files in order to begin reading them
trainImgs = fopen('train-images-idx3-ubyte','r','b');
testImgs = fopen('t10k-images-idx3-ubyte','r','b');
trainLabels = fopen('train-labels-idx1-ubyte','r','b');
testLabels = fopen('t10k-labels-idx1-ubyte','r','b');

%preparing the metadata for the training images set to be read
trainmagicnum = fread(trainImgs,1,'int32');
trainCount = fread(trainImgs,1,'int32');
trainW = fread(trainImgs,1,'int32');
trainH = fread(trainImgs,1,'int32');

%preparing the metadata for the test images set
testmagicnum = fread(testImgs,1,'int32');
testCount = fread(testImgs,1,'int32');
testW = fread(testImgs,1,'int32');
testH = fread(testImgs,1,'int32');

%preparing the metadata for the training label set
trainlabelmagicnum = fread(trainLabels,1,'int32');
trainlabelcount = fread(trainLabels,1,'int32');

%preparing the metadata for the test label set
testlabelmagicnum = fread(testLabels,1,'int32');
testLabelCount = fread(testLabels,1,'int32');


trainCount=10000; %only using the first 10000 training images
toBeTested=100; %number of test images used

% arranging the set of training images in a 784 X 60000 size matrix and
% training labels in a 1 X 60000 size matrix
imgTrainArray = zeros(trainW*trainH,trainCount);
labelTrainArray= zeros(1,trainCount);
for i=1:1:trainCount
    imgTrainArray(:,i)=fread(trainImgs,[trainW*trainH,1],'uint8');
end
labelTrainArray(1,:)=fread(trainLabels,[1,trainCount],'uint8');

% arranging the set of testing images in a 784 X 10000 size matrix and
% training labels in a 1 X 10000 size matrix
testLabelArray=fread(testLabels,[1,toBeTested],'uint8'); %label
testImgsArray=zeros(testW*testH,toBeTested);
for i=1:1:toBeTested
    testImgsArray(:,i)=fread(testImgs,[testW*testH,1],'uint8');
end



%%% TESTING images and getting percentage error%%%
incorrectCount=0; %counts incorrect comparisons of votes and labels
tot_toc = 0; %variable to allow display of duration of for loop

%%Vary K and plot K vs percentage error %%
%K value
K=13;
disp('Initiating testing');
for k=1:1:toBeTested %for all of the images that will be tested
    tic
    %matrix stores distance between test img and all training imgs used
    scalars=zeros(trainCount,2);  
    for j=1:1:trainCount %for all training images
        sum=0;
        for i=1:1:testW*testH %for all features (pixels)
            %add up (sum of differences)^2 between train and test
            sum=sum+double((imgTrainArray(i,j)-testImgsArray(i,k))^2);
        end
        scalars(j,1)=sqrt(sum);% store distance between train and test img
        scalars(j,2)=labelTrainArray(1,j); %store train label
    end
    scalars=sortrows(scalars); % sort ascendingly according to distance
    
    %count frequency of each number
    count=zeros(10,1);
    %take first K elements in sorted array and increase corresponding frequencies in count
    for i=1:1:K
        count(scalars(i,2)+1,1)=count(scalars(i,2)+1,1)+1;
    end
    %get vote with highest frequency
    [nOfVotes,index]=max(count);
    vote=index-1;
    %check if vote is same as test img label
    if(vote~=testLabelArray(1,k))
        incorrectCount=incorrectCount+1; %if incorrect increase count to calculate error
    end
    tot_toc = DisplayEstimatedTimefLoop( tot_toc+toc, k, toBeTested-1 ); %displays ETA
end
percentError=double(incorrectCount/toBeTested)*100 %result is 5 percent error for 100 test imgs




