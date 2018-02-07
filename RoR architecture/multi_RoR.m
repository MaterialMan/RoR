% Make sure files are on the path
addpath(genpath('\RoR-master-Git\'))

dataSet = 'Laser'
resType = 'RoR';
maxMinorUnits=100;
maxMajorUnits=1;
evolve_RoR


%% 
clear
rng(10,'twister');
dataSet = 'NARMA10'
resType = 'RoR_IA';
maxMinorUnits=25;
maxMajorUnits=4;

%Evolutionary parameters
numTests = 10;
popSize =15;           
numEpoch = 100;
numMutate = 0.3; 
deme = popSize-1;   
recRate = 0.4; 
rankedFitness = 0;
startFull = 1;
leakOn = 1;
genPrint = 2;

%Run 
evolve_RoR