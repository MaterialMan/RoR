function [esnMinor,esnMajor] = mutateLoser_hyper(esnMinor,esnMajor,loser,pos)

if ~isempty(esnMinor(loser,pos).nInternalUnits)
    
    mutateType = sum(rand >= cumsum([0.2,0.2,0.2,0.2,0.2]));
    switch(mutateType)
        case 0
            esnMinor(loser,pos).spectralRadius = 2*rand;
            esnMinor(loser,pos).internalWeights = esnMinor(loser,pos).spectralRadius * esnMinor(loser,pos).internalWeights_UnitSR;
        case 1
            esnMinor(loser,pos).inputScaling = 2*rand-1;
        case 2
            esnMinor(loser,pos).leakRate = rand;
        case 3
            esnMinor(loser,pos).inputShift = 2*rand-1;
        case 4
            if esnMajor(loser).nInternalUnits > 1
                pos2 = randi([1 esnMajor(loser).nInternalUnits]);
                esnMajor(loser).interResScaling{pos,pos2} = 2*rand-1;  
            end
    end
    
end