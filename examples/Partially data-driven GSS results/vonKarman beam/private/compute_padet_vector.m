function [Pade,PadeD,PadeN,Coeff_mat] = compute_padet_vector(A_Taylor,epsilon,A_coeff,B_coeff,order,dim)
%%
% Storing the coefficients in suitable format for function output.
% They are stored in lexicographic ordering of the multi-indices
%%
% A_0_out = repmat(struct('coeffs',[],'ind',[]),1,order);

PadeN =0;
PadeD = 1;
Taylor = 0;
Coeff_mat = [];
Coeff_mat = [ones(max(size(B_coeff(1,:))),1),Coeff_mat];
for indj = 1:order 
%     Taylor = Taylor + (A_Taylor(indj).coeffs(dim,:)).*epsilon^indj;
    PadeN = PadeN + A_coeff(indj,:).*epsilon^indj;
    PadeD = PadeD + B_coeff(indj).*epsilon^indj;
    Coeff_mat = [B_coeff(indj,:).',Coeff_mat];
end 
% k_list = find(abs(PadeN)<tol & abs(PadeD)<tol);

Pade = PadeN./PadeD;
% one1 = PadeD(1:end-1)-PadeD(2:end);
% check = find(abs(one1)>10);
% for ind = 1:3:max(size(check)) 
% Pade(check(ind):check(ind+2)) = NaN;
% end

Compute_diff = abs(Taylor - Pade);


% Pade(k_list) = NaN;
end