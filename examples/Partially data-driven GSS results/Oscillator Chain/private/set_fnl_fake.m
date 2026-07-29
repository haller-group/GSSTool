function [fnl_multi]  = set_fnl_fake(fnlTensor)
%Sets second order nonlinear force in multi-index format
d   = length(fnlTensor) + 1;

fnl_multi = repmat(struct('coeffs',[],'ind',[]),1,d-1);

for j = 2:d
    if isempty(fnlTensor{j-1}) || nnz(fnlTensor{j-1}) == 0

    else
        sizej = fnlTensor{j-1}.size;
        subsj = fnlTensor{j-1}.subs;
        valsj = fnlTensor{j-1}.vals;
        tmp = tensor_to_multi_index(sptensor(subsj,valsj,sizej));
        fnl_multi(j-1).coeffs = tmp.coeffs;
        fnl_multi(j-1).ind = tmp.ind;
    end
end
end