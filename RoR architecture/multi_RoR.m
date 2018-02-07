% Make sure files are on the path
addpath(genpath('\RoR-master-Git\'))
 
clear
rng(10,'twister');
dataSet = 'NARMA10'
resType = 'RoR_IA';
maxMinorUnits=25;
maxMajorUnits=8;

%Evolutionary parameters
numTests = 10;
popSize =15;           
numEpoch = 250;
numMutate = 0.3; 
deme = popSize-1;   
recRate = 0.4; 
rankedFitness = 0;
startFull = 0;
leakOn = 1;
genPrint = 2;

%Run 
evolve_RoR