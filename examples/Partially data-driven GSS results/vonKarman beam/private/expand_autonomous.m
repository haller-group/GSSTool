function S = expand_autonomous(W0,n, p)
S = sparse(n,1);
% expand autonomous coefficients
    for k = 1:length(W0)
%         S =  S + real(expand_multiindex(W0{k},p));
        S =  S + real(expand_multiindex(W0(k),p));
    end

end