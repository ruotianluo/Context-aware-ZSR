%%-- 11/13/2018 10:49:34 PM --%%
addpath(pwd); savepath;
addpath(genpath('toolbox'))
edgeBoxesDemo
fl = textread('vg_test_list.txt', '%s')
numel(fl)
for i=1:numel(fl)
    fl(i) = strcat('VG_100K_2/',fl(i));
end
fl(1)
model=load('models/forest/modelBsds'); model=model.model;
opts = edgeBoxes
opts.minScore = .07
model.opts.multiscale=1
tic, bbs=edgeBoxes(fl,model,opts); toc
save 'bbs.mat', bbs
