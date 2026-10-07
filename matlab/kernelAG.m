function G = kernelAG(U, V)
%KERNELAG  Adaptive Gaussian (AG) kernel (Eq. 20) for fitcsvm / templateSVM.
%
%   G = kernelAG(U, V)
%       U : m×p,  V : n×p   (rows are observations, columns are features)
%       G : m×n Gram block, G(i,j) = K_AG(U(i,:), V(j,:))
%
%   Reference:
%     Elen, A., Baş, S., & Közkurt, C. (2022). An Adaptive Gaussian Kernel
%     for Support Vector Machine. Arabian Journal for Science and
%     Engineering, 47, 10579–10588. doi:10.1007/s13369-022-06654-3
%
%   Definition in the paper:
%     K_AG(u,v) = exp( −(‖u−v‖² − δ + Ω) / δ )                         (Eq. 16)
%     K_AG(u,v) = exp( −(‖u−v‖² − δ + Ω) / (δ + ε) )                   (Eq. 18)
%     ξ         = ‖u−v‖² − δ                                           (Eq. 19)
%     K_AG(u,v) = exp( −(ξ + Ω) / (δ + ε) )                            (Eq. 20)
%
%   δ, Ω and ε come from kernelAG_setup(Xtrain) (Eq. 16–18). Since Ω = δ,
%   Eq. 20 is the Gaussian kernel (Eq. 5, 15) with γ = 1/(δ+ε), chosen
%   automatically from the data. Built-in equivalent:
%   'KernelFunction','rbf', 'KernelScale', sqrt(δ+ε).
%
%   Note: global variables are not passed to parfor workers; call
%   kernelAG_setup on each worker when running in parallel.

    global AG_PARAMS
    if isempty(AG_PARAMS)
        error('kernelAG:setup', 'Call kernelAG_setup(Xtrain) first.');
    end
    P = AG_PARAMS;

    D2 = pdist2(U, V, 'euclidean').^2;                      % ‖u−v‖², m×n
    xi = D2 - P.delta;                                      % Eq. 19
    G  = exp(-(xi + P.Omega) ./ (P.delta + P.epsilon));     % Eq. 18 / Eq. 20
end
