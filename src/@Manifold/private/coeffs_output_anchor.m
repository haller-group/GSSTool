function [A_0_out] = coeffs_output_anchor(A_0,order)
%%
% Storing the coefficients in suitable format for function output.
% They are stored in lexicographic ordering of the multi-indices
%%
A_0_out = repmat(struct('coeffs',[],'ind',[]),1,order);

l = 1;
for i = 1:order
    %create multi-indices
    if l >1
    K = sparse(sortrows(nsumk(l,i,'nonnegative')));
    else
        K=i;
    end
    % SSM coefficients with multi-indices
%     idx_A_0     = all(A_0(i).coeffs==0);
%     A_0_out(i).coeffs = A_0(i).coeffs(:,~idx_A_0,:);
%     A_0_out(i).ind    =  K(~idx_A_0,:);

   if ~isempty(A_0(i).coeffs)
    A_0_out(i).coeffs = squeeze(A_0(i).coeffs(:,1,1,:));
    A_0_out(i).ind    =  K;
   else
    A_0_out(i).coeffs = A_0(i).coeffs;
    A_0_out(i).ind    =  K;
   end     
    % Reduced dynamics coefficients with multi-indices
%     [~,idx_R_0] = find(R_0(i).coeffs);
%     idx_R_0     = unique(idx_R_0);
%     R_0_out(i).coeffs = R_0(i).coeffs(:,idx_R_0);
%     R_0_out(i).ind   = K(idx_R_0,:);    

end
end