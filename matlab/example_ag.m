%% Adaptive Gaussian (AG) kernel: MATLAB example on the three KEEL datasets
% Reference: Elen, Baş & Közkurt (2022), Arab J Sci Eng 47:10579–10588.
%            doi:10.1007/s13369-022-06654-3
%
% Protocol as in Section 5 of the paper: SMO solver, raw data
% ('Standardize', false), stratified 10-fold cross-validation.
% AG (Eq. 20) is compared with the built-in Gaussian/RBF kernel at γ = 1.

datasets = {'dermatology', 'vehicle', 'haberman'};
dataDir  = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
k = 10;

for d = 1:numel(datasets)
    [X, Y] = loadKeel(fullfile(dataDir, [datasets{d} '.dat']));
    rng(0);
    cv = cvpartition(Y, 'KFold', k);
    accAG = zeros(k, 1);  accRBF = zeros(k, 1);

    for f = 1:k
        Xtr = X(training(cv, f), :);  Ytr = Y(training(cv, f));
        Xte = X(test(cv, f), :);      Yte = Y(test(cv, f));

        % Eq. 16–18: δ, Ω, ε from this fold's training data only
        kernelAG_setup(Xtr);

        tAG = templateSVM('Solver', 'SMO', 'KernelOffset', 0, ...
                          'KernelFunction', 'kernelAG', ...      % Eq. 20
                          'Standardize', false);
        accAG(f) = 1 - loss(fitcecoc(Xtr, Ytr, 'Learners', tAG), Xte, Yte);

        % Gaussian/RBF with γ = 1 (KernelScale = 1/sqrt(γ) = 1), Eq. 15
        tRBF = templateSVM('Solver', 'SMO', 'KernelOffset', 0, ...
                           'KernelFunction', 'rbf', 'KernelScale', 1, ...
                           'Standardize', false);
        accRBF(f) = 1 - loss(fitcecoc(Xtr, Ytr, 'Learners', tRBF), Xte, Yte);
    end
    fprintf('%-12s  AG: %.4f   RBF (γ=1): %.4f\n', datasets{d}, mean(accAG), mean(accRBF));
end

%% Validity checks (Haberman, full data)
[X, ~] = loadKeel(fullfile(dataDir, 'haberman.dat'));
P = kernelAG_setup(X);
G = kernelAG(X, X);
fprintf('Asymmetry (Frobenius) : %.2e\n', norm(G - G', 'fro'));
fprintf('Smallest eigenvalue   : %.2e\n', min(eig((G + G') / 2)));
fprintf('Chunk difference      : %.2e\n', abs(kernelAG(X(1,:), X(2,:)) - G(1,2)));
fprintf('Diff. to Eq. 15       : %.2e\n', max(abs(G - exp(-P.gamma * pdist2(X, X).^2)), [], 'all'));
