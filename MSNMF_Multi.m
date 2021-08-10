function [U_final, V_final] = MSNMF_Multi(X, k, W1,W2,W3,options)
% Multiple graph regularized semi-supervised nonngeative matrix factorizaition (MSNMF 2021)
% where
%   X (mFea x nSmp) data matrix
% Notation:
% k ... number of hidden factors
% W1,W2,W3 ... weight matrix of the affinity graph after constraint propagation algorithm 
%
% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*V'
%
maxIter = options.maxIter;
alpha = options.alpha;
[nfea,nSmp] = size(X);

if alpha < 0
    alpha = 0;
end
%%%%% L, W, D %%%%%%
DCol1 = full(sum(W1,2));
D1 = spdiags(DCol1,0,speye(size(W1,1)));
L1 = D1 - W1;
DCol2 = full(sum(W2,2));
D2 = spdiags(DCol2,0,speye(size(W2,1)));
L2 = D2 - W2;
DCol3 = full(sum(W3,2));
D3 = spdiags(DCol3,0,speye(size(W3,1)));
L3 = D3 - W3;
%%% initial values U, V %%%%
U = abs(rand(nfea,k));
V = abs(rand(nSmp,k));
[U,V] = NormalizeUV(U, V);
b1 = 1/3;b2 = 1/3;b3 = 1/3;
nIter = 0;
while nIter < maxIter
    % ===================== update U ========================
    U = U.*(X*V./max(U*V'*V,1e-10)); 
    % ===================== update V ========================
    add1 = X'*U + alpha*(b1*W1*V+b2*W2*V+b3*W3*V);
    add2 = V*U'*U + alpha*(b1*D1*V+b2*D2*V+b3*D3*V);
    V = V.*(add1./max(add2,1e-10));
    clear add1 add2;
    % ===================== update weight parameters ========================
    b1 = 1./(2*sqrt(trace(V'*L1*V)));b2 = 1./(2*sqrt(trace(V'*L2*V)));
    b3 = 1./(2*sqrt(trace(V'*L3*V)));
    b1 = b1./(b1+b2+b3);b2 = b2./(b1+b2+b3);b3 = b3./(b1+b2+b3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    nIter = nIter + 1;
end
U_final = U;
V_final = V;
[U_final,V_final] = NormalizeUV(U_final, V_final);

%==========================================================================

function [U, V] = NormalizeUV(U, V)
    Norm = 2;
    NormV = 0;
    Kcluster = size(U,2);
    
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,Kcluster,Kcluster);
            U = U*spdiags(norms,0,Kcluster,Kcluster);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,Kcluster,Kcluster);
            V = V*spdiags(norms,0,Kcluster,Kcluster);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,Kcluster,Kcluster);
            U = U*spdiags(norms,0,Kcluster,Kcluster);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,Kcluster,Kcluster);
            V = V*spdiags(norms,0,Kcluster,Kcluster);
        end
    end
