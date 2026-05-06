clc; clear; close all;

load('Dataset\HighwayI.mat');
dataset_name='HighwayI';
% load('CAVIAR1.mat')
% dataset_name='CAVIAR1';

X = Data;
[~, ~, n3] = size(X);
[L,S,iter] = TRPCA_DWT(X);

saveRoot = fullfile('TRPCA-DWT-main\Output', dataset_name);
if ~exist(fullfile(saveRoot, 'L'), 'dir'), mkdir(fullfile(saveRoot, 'L')); end
if ~exist(fullfile(saveRoot, 'S'), 'dir'), mkdir(fullfile(saveRoot, 'S')); end

for i = 1:n3
    L_img_rgb = zeros(H, W, 3);
    S_img_rgb = zeros(H, W, 3);
    for j = 1:3
        L_img_rgb(:,:,j) = convertVector2Mat(L(:,j,i), H, W);
        S_img_rgb(:,:,j) = convertVector2Mat(S(:,j,i), H, W);
    end
    imwrite(uint8(min(max(L_img_rgb, 0), 255)), fullfile(saveRoot, 'L', sprintf('%05d.jpg', i)));
    imwrite(uint8(255 * mat2gray(abs(S_img_rgb))), fullfile(saveRoot, 'S', sprintf('%05d.jpg', i)));
end
