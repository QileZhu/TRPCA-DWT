%MAIN_BACKGROUND_MODEL  Demo of TRPCA-DWT for video background modeling.
%
%   This script loads a sample video tensor from the Dataset folder, applies
%   TRPCA-DWT, and saves the recovered low-rank background frames and sparse
%   foreground frames into the Output folder.
%
%   Required files:
%       Dataset/HighwayI.mat or Dataset/IBMtest2.mat
%
%   The MAT file is expected to contain:
%       Data - input tensor arranged as HW x 3 x T
%       H    - frame height
%       W    - frame width
%
%   To run:
%       1. Set the MATLAB current folder to this repository root.
%       2. Run main_background_model.m.

clc; clear; close all;

% Add all subfolders so that helper functions can be found.
addpath(genpath(pwd));

% Select a sample dataset.
load(fullfile('Dataset','HighwayI.mat'));
dataset_name = 'HighwayI';
% load(fullfile('Dataset','IBMtest2.mat'));
% dataset_name = 'IBMtest2';

% Run TRPCA-DWT. L is the low-rank background component and S is the sparse
% foreground/corruption component.
X = Data;
[~, ~, n3] = size(X);
[L,S,iter] = TRPCA_DWT(X); %#ok<NASGU>

% Create output folders.
saveRoot = fullfile('Output', dataset_name);
if ~exist(fullfile(saveRoot, 'L'), 'dir'), mkdir(fullfile(saveRoot, 'L')); end
if ~exist(fullfile(saveRoot, 'S'), 'dir'), mkdir(fullfile(saveRoot, 'S')); end

% Convert vectorized RGB channels back to images and save each frame.
for i = 1:n3
    L_img_rgb = zeros(H, W, 3);
    S_img_rgb = zeros(H, W, 3);
    for j = 1:3
        L_img_rgb(:,:,j) = convertVector2Mat(L(:,j,i), H, W);
        S_img_rgb(:,:,j) = convertVector2Mat(S(:,j,i), H, W);
    end

    % Low-rank component is saved directly after clipping to [0,255].
    imwrite(uint8(min(max(L_img_rgb, 0), 255)), ...
        fullfile(saveRoot, 'L', sprintf('%05d.jpg', i)));

    % Sparse component is visualized by absolute value and rescaling.
    imwrite(uint8(255 * mat2gray(abs(S_img_rgb))), ...
        fullfile(saveRoot, 'S', sprintf('%05d.jpg', i)));
end

fprintf('TRPCA-DWT finished on %s. Results saved to %s.\n', dataset_name, saveRoot);
