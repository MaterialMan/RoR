%% Select Data Script: Generate task data sets and split data
% Notes:
% - All data normalised between [-0.5 0.5]
% - Incomplete, but most datasets should work.

function [trainInputSequence,trainOutputSequence,valInputSequence,valOutputSequence,...
    testInputSequence,testOutputSequence,nForgetPoints,errType,queueType] = selectDataset(inputData, xvalDetails, extraIn)

scurr = rng;
temp_seed = scurr.Seed;

rng(1,'twister');

%set default distribution
if (nargin<3)
    extraIn = [];
end

%Default: standard 3-way split
xvalDetails.kfoldType = 'standard';%
xvalDetails.kfold =[];
xvalDetails.kfoldSize =[];

switch inputData

    %% Chaotic systems
    case 'NARMA10' %input error 4 - good task
        errType = 'NRMSE';
        queueType = 'simple';
        nForgetPoints =100; 
        sequenceLength = 8000; 
         train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        [inputSequence,outputSequence] = generate_new_NARMA_sequence(sequenceLength,10);
           fprintf('NARMA task - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
              
    case 'NARMA20' %input error 4 - good task
        errType = 'NMSE';
        queueType = 'simple';
        nForgetPoints =100; 
        sequenceLength = 8000; 
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        [inputSequence,outputSequence] = generate_new_NARMA_sequence(sequenceLength,20);
               fprintf('NARMA task - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
    case 'NARMA30' %input error 4 - good task
        errType = 'NMSE';
        queueType = 'simple';
        nForgetPoints =100; 
        sequenceLength = 8000; 
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        [inputSequence,outputSequence] = generate_new_NARMA_sequence(sequenceLength,30);
               fprintf('NARMA 30 task - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
    case 'NARMA40' %input error 4 - good task
        errType = 'NMSE';
        queueType = 'simple';
        nForgetPoints =100; 
        sequenceLength = 8000; 
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        [inputSequence,outputSequence] = generate_new_NARMA_sequence(sequenceLength,40);
               fprintf('NARMA 30 task - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
    case 'NonLinearMap&Memory' %Taken from Rodan& Tino https://www.cs.bham.ac.uk/~pxt/PAPERS/esn_jumps.pdf
        errType = 'NMSE';
        queueType = 'simple';
        nForgetPoints =100;
        sequenceLength= 8000;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;

        %create input
        s = -0.8+rand(sequenceLength,1)*1.6;
        outputNum = 1;
        beta = zeros(sequenceLength);
        %calculate all outputs - the degree of difficulty increases with
        %delay and nonlinearity
        for d = 1:15
            for p = 1:10
                for t = d+2:sequenceLength%-d+1
                    beta(t-d) = dot(s(t-d),s(t-d-1));
                    y(outputNum,t) = dot(sign(beta(t-d)),abs(beta(t-d)).^p);
                end
                outputNum = outputNum+1;
            end
        end
        
        inputSequence = s;
        outputSequence =y';
        
    case 'HenonMap' % input error > 1 - good task
        queueType = 'simple';
        errType = 'NMSE';
        nForgetPoints =100;
        sequenceLength= 8000; 
        stdev = 0.04;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
               [inputSequence,outputSequence] = generateHenonMap(sequenceLength,stdev);
               
    case 'Lorenz' % input error = 0.185 - bad task!
        queueType = 'simple';
        errType = 'NRMSE';
        nForgetPoints =100;
        sequenceLength= 4000;
        train_fraction=0.6;    val_fraction=0.2;    test_fraction=0.2;
        ahead = 1;
        [t, x, y, z]=createLorenz(10,28,8/3,100);
        inputSequence = x(1:sequenceLength-ahead);
        outputSequence = x(ahead+1:sequenceLength);
        
        %% Time-series
    case 'IPIX' % good task
        errType = 'IPIX';
        queueType = 'Weighted';
        nForgetPoints =100;
        sequenceLength = 2000; 
        train_fraction=0.4;    val_fraction=0.25;    test_fraction=0.35;   %val and test are switched later so ratios need to be swapped             
        
        % IPIX radar task
        %load hiIPIX.txt
        load loIPIX.txt
        fprintf('IPIX task - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))

        inputSequence = loIPIX(1:sequenceLength,:);
        ahead = 1;
        outputSequence =[];
        for i = 1:10
                     outputSequence = [outputSequence [inputSequence(ahead+1:end,:); zeros(ahead,2)]];
                      ahead = ahead+1;
        end
        
        fprintf('Low IPIX task. \n Started at %s \n',datestr(now, 'HH:MM:SS'))
        
        
    case 'Laser' % good task
        queueType = 'simple';
        errType = 'NMSE';
        % Sante Fe Laser generator task
        nForgetPoints =100;
        sequenceLength = 8000;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
                
        ahead = 1;
        data = laser_dataset;  %checkout the list at http://uk.mathworks.com/help/nnet/gs/neural-network-toolbox-sample-data-sets.html
        data = cell2mat(data(:,1:sequenceLength+ahead));
        inputSequence = data(1:end-ahead)';
        outputSequence = data(ahead+1:end)';

        fprintf('Laser task TSP - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        

    case 'Sunspot' % good task but not sure about dataset- problem with dividing set
        queueType = 'simple';
        errType = 'NRMSE';
        % Sunspot task - needs proper dataset separation
        nForgetPoints =30;
        sequenceLength = 3198;
        train_fraction= 2046/sequenceLength;    val_fraction=512/sequenceLength;    test_fraction=640/sequenceLength;
        
        ahead = 1;        
        load sunspot.txt %solar_dataset;  %checkout the list at http://uk.mathworks.com/help/nnet/gs/neural-network-toolbox-sample-data-sets.html
        data = sunspot(1:sequenceLength+ahead,4);       
        inputSequence = data(1:end-ahead);
        outputSequence = data(ahead+1:end);
                
        fprintf('Sunspot task TSP - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
        %mulit-in-out timeseries
        
        
    case 'secondOrderTask' %best 3.61e-3
        queueType = 'simple';
        errType = 'NMSE';

        nForgetPoints =50;
        sequenceLength = 700;
        train_fraction= 300/sequenceLength;    val_fraction=100/sequenceLength;    test_fraction=300/sequenceLength;       
        u = rand(sequenceLength,1)/2;
        y = zeros(sequenceLength,1);
        for i = 3:sequenceLength
            y(i) = 0.4*y(i-1)+0.4*y(i-1)*y(i-2)+0.6*(u(i).^2) + 0.1;
        end
        inputSequence = u;
        outputSequence = y;
        
    case 'pollution_dataset'
        
        load pollution_dataset
        
        %single in-out timeseries
    case 'valve_dataset'
        
        %single in-out timeseries
    case 'exchanger_dataset'
        
    case 'OilPrice' % input error 0.8308 1.0698 - good task 
        queueType = 'simple';
        errType = 'NRMSE';
        % Oil price task - needs proper dataset separation
        nForgetPoints =10;
        sequenceLength = 179;
        train_fraction=0.5;    val_fraction=0.25;    test_fraction=0.25;
        
        ahead = 1;
        data = oil_dataset;  %checkout the list at http://uk.mathworks.com/help/nnet/gs/neural-network-toolbox-sample-data-sets.html
        data = cell2mat(data(:,1:sequenceLength+ahead));
        inputSequence = data(:,1:end-ahead)';
        outputSequence = data(:,ahead+1:end)';
        
        fprintf('Oil price task TSP - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
    case 'HeatExchange' % input error high - good task
        
        queueType = 'simple';
        errType = 'NRMSE';
        % Oil price task - needs proper dataset separation
        nForgetPoints =100;
        sequenceLength = 2000;
        train_fraction=0.5;    val_fraction=0.25;    test_fraction=0.25;
        
        ahead = 1;
        data = exchanger_dataset;  %checkout the list at http://uk.mathworks.com/help/nnet/gs/neural-network-toolbox-sample-data-sets.html
        data = cell2mat(data(:,1:sequenceLength+ahead));
        inputSequence = data(:,1:end-ahead)';
        outputSequence = data(:,ahead+1:end)';
        
        fprintf('Heat Exchange task TSP - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        
        %% Harmonic systems
    case 'WaveGen'
        queueType = 'simple';
        errType = 'NRMSE';
        %Wave generator task
        nForgetPoints =1000;
        sequenceLength = 3600; %taken as 9000 for training from Stieg's ASN paper
        train_fraction=0.33333334;    val_fraction=0.33333334;    test_fraction=0.33333333;
        
        freq = 100;
        fprintf('Wave Generator - 64 electrode test: %s \n',datestr(now, 'HH:MM:SS'))
        fprintf('Freq: %d Hz\n',freq);
        scanFreq = 20000; %per channel
        step = 1/scanFreq;
        t = 0:step:1-step;
        amplitude = 1;
        
        % sinewave input
        inputSequence(:,1) = amplitude*sin(2*pi*freq*t);
        outputSequence(:,1) = amplitude*sawtooth(2*pi*freq*t);
        outputSequence(:,2) = amplitude*cos(2*pi*freq*t);
        outputSequence(:,3) = amplitude*square(2*pi*freq*t);
        outputSequence(:,4) = amplitude*sin(2*pi*(freq*2)*t);
        if freq >100
            inputSequence = inputSequence(1:sequenceLength,:);
            outputSequence = outputSequence(1:sequenceLength,:);
        end
        
    case 'SignalClassification'
         errType = 'NMSE';
        queueType = 'simple';
        nForgetPoints =100;        
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        
        freq = 1000;
        fprintf('Signal Classification: \n',datestr(now, 'HH:MM:SS'))
        fprintf('Freq: %d Hz\n',freq);
        scanFreq = 20000; %per channel
        step = 1/scanFreq;
        t = 0:step:1-step;
        amplitude = 1;
        sequenceLength = 6000;
        period = 20;
        
        % sinewave input
        inputSequence(:,1) = amplitude*sin(2*pi*freq*t);
        inputSequence(:,2) = amplitude*square(2*pi*freq*t);
        
        cnt = 1; sinInput =[];squareInput=[];
        for i = 0:period:sequenceLength-period
            sinInput(cnt,i+1:i+period) = inputSequence(i+1:i+period,1);
            squareInput(cnt,i+1:i+period) = inputSequence(i+1:i+period,2);
            cnt = cnt +1;
        end
        
        combInput = zeros(sequenceLength,1); combOutput= ones(sequenceLength,2)*-1;
        for i = 1:sequenceLength/period
            if round(rand)
                combInput = combInput+sinInput(i,:)';
                combOutput((i*period)-period+1:i*period,1) =  ones(period,1);
            else
                combInput = combInput+squareInput(i,:)';
                combOutput((i*period)-period+1:i*period,2) =  ones(period,1);
            end
        end
        
        inputSequence = combInput;
        outputSequence = combOutput;
        figure
        subplot(2,1,1)
        plot(inputSequence(1:350))
        subplot(2,1,2)
        plot(outputSequence(1:350,:))
        
    case 'MSO' %not really applicable because the task is generative
        errType = 'NRMSE';
        queueType = 'simple'; %?
        nForgetPoints =100;
        sequenceLength= 2400;
        train_fraction=0.333333;    val_fraction=0.333334;    test_fraction=0.3333334;
        
        for t = 1:sequenceLength
            u(1,t) = sin(0.2*t)+sin(0.311*t);
            u(2,t) = sin(0.2*t)+sin(0.311*t)+sin(0.42*t);
%             u(3,t) = sin(0.2*t)+sin(0.311*t)+sin(0.42*t)+sin(0.51*t);
%             u(4,t) = sin(0.2*t)+sin(0.311*t)+sin(0.42*t)+sin(0.51*t)+sin(0.74*t);
%             u(5,t) = sin(0.2*t)+sin(0.311*t)+sin(0.42*t)+sin(0.51*t)+sin(0.63*t)+sin(0.74*t)+sin(0.85*t)+sin(0.95*t);
        end
        %predictor - not sure what predictor value is best
        ahead = 10;
        inputSequence = u(:,1:end-ahead)';
        outputSequence = u(:,ahead+1:end)';
     
        %% Pattern Recognition - using PCA to reduce dimensions maybe very useful
    case 'handDigits'
        errType = 'OneVsAll';
        queueType = 'Weighted';
        xvalDetails.kfoldType = 'Randperm';
        nForgetPoints =10;
        train_fraction=0.8;    val_fraction=0.1;    test_fraction=0.1;
        datasetLength = 5000; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        
        load('\Datasets\handDigits.mat');
        inputSequence = X;
        for i = 1:10
            outputSequence(:,i) = y==i;
        end
        
        
    case 'JapVowels' %(12: IN, 9:OUT - binary ) - input only 83% accuracy!  Train:0.2288  Test:0.1863
        errType = 'OneVsAll'; %Paper: Optimization and applications of echo state networks with leaky- integrator neurons
        queueType = 'Weighted';
        % Nine male speakers uttered two Japanese vowels /ae/ successively.
        % For each utterance, with the analysis parameters described below, we applied
        % 12-degree linear prediction analysis to it to obtain a discrete-time series
        % with 12 LPC cepstrum coefficients. This means that one utterance by a speaker
        % forms a time series whose length is in the range 7-29 and each point of a time
        % series is of 12 features (12 coefficients).
        % The number of the time series is 640 in total. We used one set of 270 time series for
        % training and the other set of 370 time series for testing.
        nForgetPoints =100;
       
        [trainInputSequence,trainOutputSequence,testInputSequence,testOutputSequence] = readJapVowels();
        inputSequence = [trainInputSequence; testInputSequence];
        outputSequence = [trainOutputSequence; testOutputSequence];
        train_fraction=size(trainInputSequence,1)/9961;    val_fraction=(size(testInputSequence,1)/9961)*0.1;    test_fraction=(size(testInputSequence,1)/9961)*0.9;
        
    case 'NIST-64_IsolatedSpokenDigit' %Paper: Reservoir-based techniques for speech recognition
        errType = 'OneVsAll_NIST';
        queueType = 'Weighted';
        nForgetPoints =150;
        xvalDetails.kfold = 5;
        xvalDetails.kfoldSize = 150;
        xvalDetails.kfoldType = 'standard';
        train_fraction=0.7;    val_fraction=0.15;    test_fraction=0.15; %meaningless
        
        y_list = [];
        u_list = [];
        lens = [];
        
        for i = 1:5
            l = [ 1 2 5 6 7];
           for j = 1:10
            for n = 0:9
                u_z = zeros(77,xvalDetails.kfoldSize);
                u = load(strcat('s',num2str(l(i)),'_u',num2str(j),'_d',num2str(n)));
                u_z(:,1:size(u.spec,2)) = u.spec;
                
                y = zeros(10,size(u_z,2))-1;
                y(n+1,:) = ones(1,size(u_z,2));
                u_list = [u_list u_z];
                lens = [lens size(u_z,2)];
                y_list = [y_list y];
            end
           end
        end
        
        inputSequence = u_list';
        outputSequence = y_list';
        
        %% Classification
    case 'simpleClass' %(2: IN, 4:OUT - binary ) - plot to see simple 4 clusters:- scatter(inputSequence(:,1),inputSequence(:,2)) - solvable just with input
        errType = 'confusion';
        queueType = 'Weighted';
        nForgetPoints =100;
        train_fraction=0.5;    val_fraction=0.25;    test_fraction=0.25;
        datasetLength = 1000; 
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;%
        
        [inputSequence, outputSequence] =  simpleclass_dataset; 
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'NonChanEq28db' % (1:in, 1:out) note: needs to be done 7 times across different noise levels
        errType = 'SER';
        queueType = 'simple'; %input alone error = 0.09
        nForgetPoints =100;
        sequenceLength = 10000; 
        train_fraction=0.6;    val_fraction=0.2;    test_fraction=0.2;
        xvalDetails.kfold = sequenceLength;
        xvalDetails.kfoldSize = sequenceLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 'standard';% 0 =
        
        [inputSequence, outputSequence] = NonLinear_ChanEQ_data(sequenceLength);
        inputSequence = inputSequence(5,:)'; %5 = 28db
        outputSequence = outputSequence';
        
    case 'NonChanEq12db' % (1:in, 1:out) note: needs to be done 7 times across different noise levels
        errType = 'SER';
        queueType = 'simple'; % input alone error= 0.1914
        nForgetPoints =100;
        sequenceLength = 10000; 
        train_fraction=0.6;    val_fraction=0.2;    test_fraction=0.2;
        xvalDetails.kfold = sequenceLength;
        xvalDetails.kfoldSize = sequenceLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 'standard';
        
        [inputSequence, outputSequence] = NonLinear_ChanEQ_data(sequenceLength);
        inputSequence = inputSequence(1,:)'; %5 = 28db
        outputSequence = outputSequence';
        
    case 'NonChanEq24db' % (1:in, 1:out) note: needs to be done 7 times across different noise levels
        errType = 'SER';
        queueType = 'simple'; %input alone error = 0.091
        nForgetPoints =100;
        sequenceLength = 10000; 
        train_fraction=0.6;    val_fraction=0.2;    test_fraction=0.2;
        xvalDetails.kfold = sequenceLength;
        xvalDetails.kfoldSize = sequenceLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 'standard';
        
        [inputSequence, outputSequence] = NonLinear_ChanEQ_data(sequenceLength);
        %inputSequence = inputSequence(extraIn,:)';
        inputSequence = inputSequence(4,:)'; %5 = 28db
        outputSequence = outputSequence';
        
     case 'NonChanEqRodan' % (1:in, 1:out) error 0.999 Good task, requires memory
        errType = 'NMSE';
        queueType = 'simple'; %input alone error = 0.091
        nForgetPoints =200;
        sequenceLength = 8000; 
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        xvalDetails.kfold = sequenceLength;
        xvalDetails.kfoldSize = sequenceLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 'standard';
        
        ahead = 2;
        [inputSequence, outputSequence] = NonLinear_ChanEQ_data(sequenceLength);
        inputSequence =  inputSequence(7,:)'+30; %5 = 28db
        outputSequence = [zeros(ahead,1); outputSequence(:,1:end-ahead)']; %inputshift
        
           
    case 'Iris' %iris_dataset; (4:in, 3:out) %input alone 76% - medium task
        errType = 'OneVsAll';%'confusion';
        queueType = 'Weighted';
        nForgetPoints =10;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        datasetLength = 150;
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  iris_dataset; %iris_dataset; (4:in, 3:out)
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'Wine' %(13:in, 3:out) % error = 0 with input alone - bad task
        errType = 'confusion';
        queueType = 'Weighted';
        nForgetPoints =10;
        train_fraction=0.25;    val_fraction=0.25;    test_fraction=0.5;
        datasetLength = 178; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;%0 = randomise data
        
        [inputSequence, outputSequence] = wine_dataset; %(13:in, 3:out)
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'Banknote' %(5: In, 1: Out - binary fake/real) %accurat with just input,  error = 0.2014
        errType = 'OneVsAll';
        queueType = 'Weighted';
        nForgetPoints =50;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        datasetLength = 1372; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;%0 = randomise data
        
        load('banknotAuth.txt');
        inputSequence = banknotAuth(:,1:4);
        outputSequence(:,1) = banknotAuth(:,5);
        outputSequence(:,2) = 1-banknotAuth(:,5);
        
    case 'Cancer'  %(9: In, 2: Out -  benign (1) or malignant (2) ) - 0.0645 accurate with input 
        errType = 'OneVsAll';
        queueType = 'Weighted';
        nForgetPoints =50;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        datasetLength = 699; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  cancer_dataset; 
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'N-bitParity'
        nForgetPoints =100;
        
        %% Function fitting
    case 'simplefit'
        errType = 'NRMSE';
        queueType = 'simple';
        nForgetPoints =10;
        train_fraction=0.3;    val_fraction=0.35;    test_fraction=0.35;
        datasetLength = 94; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  simplefit_dataset;
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'Chemicalfit' %(8:IN, 1:Out) - good task!!
        errType = 'NMSE';
        queueType = 'Weighted';
        nForgetPoints =25;
        train_fraction=0.5;    val_fraction=0.25;    test_fraction=0.25;
        datasetLength = 498; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  chemical_dataset;
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'EngineFit' % 2: IN, 2: OUT - good task
        
        errType = 'NMSE';
        queueType = 'Weighted';
        nForgetPoints =100;
        train_fraction=0.25;    val_fraction=0.375;    test_fraction=0.375;
        datasetLength = 1198; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  engine_dataset;
        
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
    case 'BuildingFit' %14: IN, 3:out - good task!
        
        errType = 'NMSE';
        queueType = 'Weighted';
        nForgetPoints =100;
        train_fraction=0.5;    val_fraction=0.25;    test_fraction=0.25;
        datasetLength = 4208; %manually change dataset length for xval
        xvalDetails.kfold = datasetLength;
        xvalDetails.kfoldSize = datasetLength/xvalDetails.kfold;
        xvalDetails.kfoldType = 0;% 0 = randomise data
        
        [inputSequence, outputSequence] =  building_dataset;
        inputSequence = inputSequence';
        outputSequence = outputSequence';
        
end

%normalise all features
%[inputSequence, mu, sigma] = featureNormalize(inputSequence);

%squash
if ~strcmp(inputData,'test') || ~strcmp(inputData,'NonLinearMap&Memory') || ~strcmp(inputData,'SignalClassification') 
    for i = 1:size(inputSequence,2)
        if max(inputSequence(:,i)) ~= 0
            inputSequence(:,i) = ((inputSequence(:,i)-mean(inputSequence(:,i)))/((max(inputSequence(:,i))-min(inputSequence(:,i)))))-0.5;
        end
    end
end
        
[trainInputSequence,trainOutputSequence,valInputSequence,valOutputSequence,...
    testInputSequence,testOutputSequence]= KFoldXValidation(inputSequence,outputSequence,xvalDetails.kfold,xvalDetails.kfoldSize,xvalDetails.kfoldType,train_fraction,val_fraction,test_fraction);

% Go back to old seed
rng(temp_seed,'twister');