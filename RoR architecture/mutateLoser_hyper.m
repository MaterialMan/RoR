function [esnMinor,esnMajor] = mutateLoser_hyper(esnMinor,esnMajor,loser,pos)

%mutateType = sum(rand >= cumsum([0.15,0.15,0.1,0.1,0.1,0.2,0.2]));
if ~isempty(esnMinor(loser,pos).nInternalUnits)

%mutateType = sum(rand >= cumsum([0.3333,0.3333,0.3333]));  
mutateType = sum(rand >= cumsum([0.25,0.25,0.25,0.25])); 
switch(mutateType)
    case 0
            esnMinor(loser,pos).spectralRadius = 2*rand;
            esnMinor(loser,pos).internalWeights = esnMinor(loser,pos).spectralRadius * esnMinor(loser,pos).internalWeights_UnitSR;
    case 1
            esnMinor(loser,pos).inputScaling = 2*rand-1;
    case 2
            esnMinor(loser,pos).leakRate = rand;
    case 3
        if esnMajor(loser).nInternalUnits > 1
        true = 1;
        while(true)
            pos2 = randi([1 esnMajor(loser).nInternalUnits]);
            if pos2 ~= pos
                true =0;
            end
        end
        val = 2*rand-1;
        esnMajor(loser).interResScaling{pos,pos2} = val;
        esnMajor(loser).interResScaling{pos2,pos} = val;
        end
end

end