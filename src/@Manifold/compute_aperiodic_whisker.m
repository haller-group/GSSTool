function [W_0, R_0,timer,zk] = compute_aperiodic_whisker(obj, order)
% Invariant manifold in the autonomous system limit

% Leading-order terms
Lambda_E = obj.E.spectrum;
W_01     = obj.E.basis;

switch obj.Options.notation
    
    case 'tensor'
        % Initialize
        W_0 = cell(1,order);
        R_0 = cell(1,order);
        
        W_0{1} = sptensor(W_01);
        R_0{1} = sptensor(diag(Lambda_E));
        
        % *Outer resonance case* : issue warning
        if obj.resonance.outer.occurs
            prompt = 'Due to (near) outer resonance, the exisitence of the manifold is questionable and the underlying computation may suffer.';
            disp(prompt)
            disp('Attempting manifold computation')
        end
        
        for j = 2:order
            %recursively approximate at j-th order
            startOrderj = tic;
            [W_0{j},R_0{j},~] = cohomological_solution(obj,j,W_0,R_0,[]);
            obj.solInfo.timeEstimate(j) = toc(startOrderj);
            disp(['Manifold computation time at order ' num2str(j) ' = ' datestr(datenum(0,0,0,0,0,obj.solInfo.timeEstimate(j)),'HH:MM:SS')])
            fprintf('Estimated memory usage at order % 2i = %05.2E MB\n', j, obj.solInfo.memoryEstimate(j))
        end        
        
        for j = 1:order
            W_0{j} = tensor_to_multi_index(W_0{j});
            R_0{j} = tensor_to_multi_index(R_0{j});
        end
        W_0 = [W_0{:}];
        R_0 = [R_0{:}];
    case 'multiindex'
        
        W_0 = repmat(struct('coeffs',[]),1,order);
        R_0 = repmat(struct('coeffs',[]),1,order);
        
        % Set up system for Multi-index notation version
%         DStype = check_DStype(obj); % whether system is real or complex
        [W_0(1),R_0(1),multi_input] = coeffs_setup_aperiodic_whisker(obj,order);
        
        % *Outer resonance case* : issue warning
        if obj.resonance.outer.occurs
            prompt = 'Due to (near) outer resonance, the exisitence of the manifold is questionable and the underlying computation may suffer.';
            disp(prompt)
            disp('Attempting manifold computation')
        end
        
        for j = 2:order
            %recursively approximate at j-th order
            startOrderj = tic;
            [W_0(j).coeffs,R_0(j).coeffs,multi_input,timer] = cohomological_solution_aperiodic_whisker(obj,j,W_0,R_0,multi_input);
            obj.solInfo.timeEstimate(j) = toc(startOrderj);
            disp(['Manifold computation time at order ' num2str(j) ' = ' datestr(datenum(0,0,0,0,0,obj.solInfo.timeEstimate(j)),'HH:MM:SS')])
            fprintf('Estimated memory usage at order % 2i = %05.2E MB\n', j, obj.solInfo.memoryEstimate(j))            
        end
%         switch DStype
%             case 'real'
%                 [W_0,R_0] = coeffs_conj2lex(multi_input,order,W_0,R_0);
%             case 'complex'
%                 [W_0] = coeffs_lex2revlex(W_0,'TaylorCoeff');
%                 [R_0] = coeffs_lex2revlex(R_0,'TaylorCoeff');
%         end
        %%
        % Add multi-indices field to the coefficients field at every order
%         [W_0,R_0] = coeffs_output(W_0,R_0,order);
        [W_0,R_0] = coeffs_output_aperiodic(W_0,R_0,order);
        zk = multi_input.Z_cci;
end

