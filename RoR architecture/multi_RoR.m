% Make sure files are on the path
addpath(genpath('\RoR-master-Git\'))

dataSet = 'Laser'
resType = 'RoR';
maxMinorUnits=100;
maxMajorUnits=1;
evolve_RoR

dataSet = 'NARMA10'
resType = 'RoR_IA';
maxMinorUnits=63;
maxMajorUnits=2;
evolve_RoR-