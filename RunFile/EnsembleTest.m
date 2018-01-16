function PredLab = EnsembleTest(Directory, fileExtension)
%% For example: fileExtension = 'bmp', 'jpg' etc
%% For example: Directory = '../OCT_Data/';
%% Path of the OCT B-scans you need to Segment (In this case the images are in '.bmp' format as RGB)
Files = dir([Directory, '*.',fileExtension]);


for i=1:numel(Files)
    tic
    ID = Files(i).name;
    Dat = imread([Directory,ID]);
    if size(Dat,3)==3
        Data = single(mat2gray(rgb2gray(Dat)));
    else
        Data = single(mat2gray(Dat));
    % Fold 1
    load(../TrainedModels/NetFold1.mat),
    fnet1 = dagnn.DagNN.loadobj(net);
    fnet1.move('gpu'); fnet1.mode = 'test';
    fnet1.eval({'input', gpuArray(single(Data))});
    Pred1 = gather(fnet1.vars(fnet1.getVarIndex('prob')).value);
    % Fold 2
    load(../TrainedModels/NetFold2.mat),
    fnet2 = dagnn.DagNN.loadobj(net);
    fnet1.move('cpu'); fnet2.move('gpu'); fnet2.mode = 'test';
    fnet2.eval({'input', gpuArray(single(Data))});
    Pred2 = gather(fnet2.vars(fnet2.getVarIndex('prob')).value);
    % Fold 3
    load(../TrainedModels/NetFold3.mat),
    fnet3 = dagnn.DagNN.loadobj(net);
    fnet2.move('cpu'); fnet3.move('gpu'); fnet3.mode = 'test';
    fnet3.eval({'input', gpuArray(single(Data))});
    Pred3 = gather(fnet3.vars(fnet3.getVarIndex('prob')).value);
    % Fold 4
    load(../TrainedModels/NetFold4.mat),
    fnet4 = dagnn.DagNN.loadobj(net);
    fnet3.move('cpu'); fnet4.move('gpu'); fnet4.mode = 'test';
    fnet4.eval({'input', gpuArray(single(Data))});
    Pred4 = gather(fnet4.vars(fnet4.getVarIndex('prob')).value);
    % Fold 5
    load(../TrainedModels/NetFold5.mat),
    fnet5 = dagnn.DagNN.loadobj(net);
    fnet4.move('cpu'); fnet5.move('gpu'); fnet5.mode = 'test';
    fnet5.eval({'input', gpuArray(single(Data))});
    Pred5 = gather(fnet5.vars(fnet5.getVarIndex('prob')).value);
    % Fold 6
    load(../TrainedModels/NetFold6.mat),
    fnet6 = dagnn.DagNN.loadobj(net);
    fnet5.move('cpu'); fnet6.move('gpu'); fnet6.mode = 'test';
    fnet6.eval({'input', gpuArray(single(Data))});
    Pred6 = gather(fnet6.vars(fnet6.getVarIndex('prob')).value);
    % Fold 7
    load(../TrainedModels/NetFold7.mat),
    fnet7 = dagnn.DagNN.loadobj(net);
    fnet6.move('cpu'); fnet7.move('gpu'); fnet7.mode = 'test';
    fnet7.eval({'input', gpuArray(single(Data))});
    Pred7 = gather(fnet7.vars(fnet7.getVarIndex('prob')).value);
    % Fold 8
    load(../TrainedModels/NetFold8.mat),
    fnet8 = dagnn.DagNN.loadobj(net);
    fnet7.move('cpu'); fnet8.move('gpu'); fnet8.mode = 'test';
    fnet8.eval({'input', gpuArray(single(Data))});
    Pred8 = gather(fnet8.vars(fnet8.getVarIndex('prob')).value);
    
    Pred = (1/8)*(Pred1 + Pred2 + Pred3 + Pred4 + Pred5 + Pred6 + Pred7 + Pred8); % xpectation over ensemble
    [~,PredLab] = max(Pred, [], 3); 
    
    t = toc
    disp(['Time to segment is ',num2str(t),' seconds'])
    save([ID(1:end-4),'_Pred.mat'], 'PredLab');
end
