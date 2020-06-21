function chips = mppsk_bb_modulator1(Symbols,A,B,K,N,rg,modMethod)

nSymbol = length(Symbols);
chips = zeros(1,N*nSymbol);

if A==1
    seq0 = ones(1,N);
    seq1 = [-(B/A)*ones(1,K),ones(1,N-K)];
else
    seq0 = (A/B)*ones(1,N);
    seq1 = [-ones(1,K),(A/B)*ones(1,N-K)];
end

%%
for i=1:nSymbol
    code=Symbols(i);
    if code==0
       chips( (i-1)*N+1:i*N ) = seq0;
    else
        if modMethod==1        
           chips( (i-1)*N+1:i*N ) = [seq0(1:(code-1)*(K+rg)),seq1(1:K),seq0( code*K+(code-1)*rg+1:end )];   
        elseif modMethod==2    
           chips( (i-1)*N+1:i*N ) = [seq0(1:code*(K+rg)),seq1(1:K),seq0( (code+1)*K+code*rg+1:end )];  
        else
            chips((i-1)*N+1:i*N )=  [seq0(1:(code-1)*K),seq1(1:(1-rg)*K),seq0( (code-rg)*K+1:end )];
        end
    end
end
end

