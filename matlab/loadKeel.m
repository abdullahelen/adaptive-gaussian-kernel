function [X, Y, names] = loadKeel(path)
%LOADKEEL  Load a KEEL .dat file.
%   [X, Y, names] = loadKeel(path)
%   Numeric inputs are kept, nominal inputs are ordinal-encoded, rows with
%   missing values ('?') are dropped. Y is returned as categorical.

    lines = strtrim(splitlines(fileread(path)));
    lines = lines(~cellfun(@isempty, lines));

    isAttr = startsWith(lower(lines), '@attribute');
    names  = cellfun(@(s) strtok(s(11:end)), lines(isAttr), 'UniformOutput', false);

    iData = find(startsWith(lower(lines), '@data'), 1);
    rows  = lines(iData+1:end);
    rows  = rows(~contains(rows, '?'));
    parts = cellfun(@(s) strtrim(strsplit(s, ',')), rows, 'UniformOutput', false);
    A     = vertcat(parts{:});                   % r×(p+1) cell array

    p = size(A, 2) - 1;
    X = zeros(size(A, 1), p);
    for j = 1:p
        v = str2double(A(:, j));
        if any(isnan(v))                         % nominal attribute
            [~, ~, v] = unique(A(:, j));
        end
        X(:, j) = v;
    end
    Y     = categorical(A(:, end));
    names = names(1:p);
end
