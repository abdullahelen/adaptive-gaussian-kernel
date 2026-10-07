function P = kernelAG_setup(Xtrain)
%KERNELAG_SETUP  Computes the data-driven constants (δ, Ω, ε) of the
%   Adaptive Gaussian (AG) kernel ONCE from the training data and stores
%   them for kernelAG.
%
%   Reference:
%     Elen, A., Baş, S., & Közkurt, C. (2022). An Adaptive Gaussian Kernel
%     for Support Vector Machine. Arabian Journal for Science and
%     Engineering, 47, 10579–10588. doi:10.1007/s13369-022-06654-3
%
%   Equations from the paper (Section 4):
%     δ(·) : standard deviation function                              (Eq. 16)
%     Ω    = |min(‖u−v‖² − δ)| if min < 0, else 0                     (Eq. 17)
%     ε    : negligible number that avoids division by zero          (Eq. 18)
%     ξ    = ‖u−v‖² − δ                                              (Eq. 19)
%
%   δ and Ω are computed over ALL pairs of the training set (diagonal
%   included), following the paper's statement that the offset is
%   "applied to the entire dataset", and are then kept fixed. Whatever
%   chunks (U, V) MATLAB passes to the kernel, the same (u,v) pair always
%   gets the same value; the Gram matrix is symmetric and PSD.
%
%   Usage:
%     P = kernelAG_setup(Xtrain);   % in every CV fold, with that fold's training data
%   If 'Standardize', true is used, pass Xtrain standardized the same way.

    global AG_PARAMS

    n  = size(Xtrain, 1);
    d2 = pdist(Xtrain, 'euclidean').^2;      % ‖u−v‖² for i<j

    % --- Eq. 16: δ(‖u−v‖²) --------------------------------------------------
    % Standard deviation over every element of the n×n squared-distance
    % matrix. Off-diagonal pairs count twice, the n zero diagonal entries
    % once; computed from sums without building the matrix (N−1, as std()).
    N     = n^2;
    S1    = 2 * sum(d2);
    S2    = 2 * sum(d2.^2);
    mu    = S1 / N;
    delta = sqrt(max((S2 - N * mu^2) / (N - 1), 0));

    % --- Eq. 19: min ξ ------------------------------------------------------
    % The diagonal is included, so min‖u−v‖² = 0.
    xiMin = 0 - delta;

    % --- Eq. 17: Ω ----------------------------------------------------------
    if xiMin < 0
        Omega = abs(xiMin);
    else
        Omega = 0;
    end

    % --- Eq. 18: ε ----------------------------------------------------------
    epsilon = eps;

    P = struct('delta', delta, 'Omega', Omega, 'epsilon', epsilon, ...
               'gamma', 1 / (delta + epsilon), 'n', n);
    AG_PARAMS = P;
end
