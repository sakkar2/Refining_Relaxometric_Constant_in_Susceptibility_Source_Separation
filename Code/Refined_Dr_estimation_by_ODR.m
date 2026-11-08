%% Refined Dr estimation by orthogonal distance regression (ODR)
%
% Dr is estimated from subject-wise ROI mean values using orthogonal
% distance regression (ODR) with two models:
%
%   Model 1 (2D linear) : R2' = a*QSM + c          (deep gray matter ROIs)
%   Model 2 (3D planar) : R2' = a*QSM + b*MT + c   (deep gray + white matter ROIs)
%
% where the slope "a" corresponds to Dr (Hz/ppm) and MT is an MT-related
% map (MTsat, or F for the CSU dataset).
%
% Standard errors and 95% confidence intervals of the parameters are
% obtained by a subject-level (cluster) bootstrap.
%
% Requirements: MATLAB with the Optimization Toolbox (lsqnonlin) and the
%               Statistics and Machine Learning Toolbox (randsample, prctile).
%
% Author: Taechang Kim (@ Seoul National Univerisity)

clear; clc;

%% User settings
% Path to the subject-wise ROI mean data (.mat). Available datasets:
%   data/ROI_mean_dataset1.mat      (3T)
%   data/ROI_mean_dataset2.mat      (3T)
%   data/ROI_mean_dataset3.mat      (3T)
%   data/ROI_mean_dataset4.mat      (7T)
data_path = fullfile('data', 'ROI_mean_dataset1.mat');

% Number of deep gray matter (DGM) ROIs in the data.
% ROI layout (columns of each matrix): 1-5 = DGM (ULiege: 1-4), remaining = WM
n_dgm = 5;      % set to 4 for the dataset4

% Number of bootstrap resamples and random seed (for reproducibility)
n_boot = 2000;
seed   = 42;

%% Load data
% Each file contains QSM_mean, R2p_mean, and MT_mean:
%   [N_subject x N_ROI] ROI mean values (QSM in ppm, R2' in Hz, MT in a.u.)
load(data_path, 'QSM_mean', 'R2p_mean', 'MT_mean');

% Model 1 uses the DGM ROIs; Model 2 uses all ROIs (DGM + WM)
rois_model1 = 1:n_dgm;
rois_model2 = 1:size(QSM_mean, 2);

%% ODR fitting
out = odr_fit_bootstrap(QSM_mean, R2p_mean, MT_mean, ...
                        rois_model1, rois_model2, n_boot, seed);


%% ========================================================================
%  Local functions
%  ========================================================================

function out = odr_fit_bootstrap(QSM_mean, R2p_mean, MT_mean, ...
                                 rois_m1, rois_m2, n_boot, seed)
% Orthogonal distance regression (ODR) for
%   Model 1: R2' = a*QSM + c
%   Model 2: R2' = a*QSM + b*MT + c
% with subject-level (cluster) bootstrap standard errors.
%
% Inputs
%   QSM_mean, R2p_mean, MT_mean : [N_subject x N_ROI] ROI mean matrices
%   rois_m1, rois_m2            : ROI (column) indices for Model 1 and 2
%   n_boot                      : number of bootstrap resamples
%   seed                        : random seed
%
% Output (struct)
%   out.m1 / out.m2 with fields
%     beta    : parameter estimates ([a c] for Model 1, [a b c] for Model 2)
%     SE_boot : bootstrap standard errors
%     CI_boot : bootstrap 95% percentile CI ([N_param x 2], [lower upper])

    rng(seed);
    opts = optimoptions('lsqnonlin', 'Display', 'off', ...
                        'FiniteDifferenceType', 'forward');

    subjects = (1:size(QSM_mean, 1)).';
    p10 = [324, 0];      % initial guess for Model 1 [a c]
    p20 = [324, 1, 0];   % initial guess for Model 2 [a b c]

    % ---------- Fit to all subjects ----------
    [x1, y1]     = stack_data(QSM_mean, R2p_mean, [],      subjects, rois_m1);
    [x2, y2, f2] = stack_data(QSM_mean, R2p_mean, MT_mean, subjects, rois_m2);

    p1 = fit_model1(x1, y1, p10, opts);
    p2 = fit_model2(x2, y2, f2, p20, opts);

    % ---------- Subject-level (cluster) bootstrap ----------
    % Subjects are resampled with replacement, and all ROIs of a resampled
    % subject are kept together.
    nS = numel(subjects);
    boots1 = nan(n_boot, 2);   % [a c]
    boots2 = nan(n_boot, 3);   % [a b c]

    for b = 1:n_boot
        Sb = subjects(randsample(nS, nS, true));

        [x1b, y1b]      = stack_data(QSM_mean, R2p_mean, [],      Sb, rois_m1);
        [x2b, y2b, f2b] = stack_data(QSM_mean, R2p_mean, MT_mean, Sb, rois_m2);

        % Failed fits (rare non-convergence) are left as NaN and discarded
        try, boots1(b,:) = fit_model1(x1b, y1b, p10, opts);      catch, end
        try, boots2(b,:) = fit_model2(x2b, y2b, f2b, p20, opts); catch, end
    end

    boots1 = boots1(all(isfinite(boots1), 2), :);
    boots2 = boots2(all(isfinite(boots2), 2), :);

    % ---------- Package results ----------
    out.m1.beta    = p1(:);
    out.m1.SE_boot = std(boots1, 0, 1).';
    out.m1.CI_boot = prctile(boots1, [2.5 97.5], 1).';   % [2 x 2]

    out.m2.beta    = p2(:);
    out.m2.SE_boot = std(boots2, 0, 1).';
    out.m2.CI_boot = prctile(boots2, [2.5 97.5], 1).';   % [3 x 2]

    % ---------- Report ----------
    fprintf('\nModel 1 (2D linear)  R2'' = a*QSM + c  (orthogonal residual)\n');
    disp_row('a', out.m1, 1);
    disp_row('c', out.m1, 2);

    fprintf('\nModel 2 (3D planar)  R2'' = a*QSM + b*MT + c  (orthogonal residual)\n');
    disp_row('a', out.m2, 1);
    disp_row('b', out.m2, 2);
    disp_row('c', out.m2, 3);
    fprintf('\n');
end

function [x, y, f] = stack_data(QSM_mean, R2p_mean, MT_mean, subjects, rois)
% Vectorize the ROI mean values of the selected subjects and ROIs.
    x = reshape(QSM_mean(subjects, rois), [], 1);
    y = reshape(R2p_mean(subjects, rois), [], 1);
    if isempty(MT_mean)
        f = [];
    else
        f = reshape(MT_mean(subjects, rois), [], 1);
    end
end

function p = fit_model1(x, y, p0, opts)
% ODR fit of the line y = a*x + c (orthogonal residual)
    res = @(p) (p(1).*x + p(2) - y) ./ sqrt(1 + p(1).^2);
    p = lsqnonlin(res, p0, [], [], opts);
end

function p = fit_model2(x, y, f, p0, opts)
% ODR fit of the plane y = a*x + b*f + c (orthogonal residual)
    res = @(p) (p(1).*x + p(2).*f + p(3) - y) ./ sqrt(1 + p(1).^2 + p(2).^2);
    p = lsqnonlin(res, p0, [], [], opts);
end

function disp_row(name, m, k)
% Print the k-th parameter of model result m.
    fprintf('  %-2s = % .6g  | SE_boot = % .3g  95%% CI_boot = [% .3g, % .3g]\n', ...
        name, m.beta(k), m.SE_boot(k), m.CI_boot(k,1), m.CI_boot(k,2));
end
