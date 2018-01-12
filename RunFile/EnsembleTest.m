clc
clear
close all
%%
Files = dir('/home/deeplearning/Abhijit/OCTdata_QUT/*.bmp');


for i=1:numel(Files)
    tic
    ID = Files(i).name;
    Data = single(mat2gray(rgb2gray(imread(['/home/deeplearning/Abhijit/OCTdata_QUT/',ID]))));
    % Fold 1
    fnet1.move('gpu'); fnet1.mode = 'test';
    fnet1.eval({'input', gpuArray(single(Data))});
    Pred1 = gather(fnet1.vars(fnet1.getVarIndex('prob')).value);
    % Fold 2
    fnet1.move('cpu'); fnet2.move('gpu'); fnet2.mode = 'test';
    fnet2.eval({'input', gpuArray(single(Data))});
    Pred2 = gather(fnet2.vars(fnet2.getVarIndex('prob')).value);
    % Fold 3
    fnet2.move('cpu'); fnet3.move('gpu'); fnet3.mode = 'test';
    fnet3.eval({'input', gpuArray(single(Data))});
    Pred3 = gather(fnet3.vars(fnet3.getVarIndex('prob')).value);
    % Fold 4
    fnet3.move('cpu'); fnet4.move('gpu'); fnet4.mode = 'test';
    fnet4.eval({'input', gpuArray(single(Data))});
    Pred4 = gather(fnet4.vars(fnet4.getVarIndex('prob')).value);
    % Fold 5
    fnet4.move('cpu'); fnet5.move('gpu'); fnet5.mode = 'test';
    fnet5.eval({'input', gpuArray(single(Data))});
    Pred5 = gather(fnet5.vars(fnet5.getVarIndex('prob')).value);
    % Fold 6
    fnet5.move('cpu'); fnet6.move('gpu'); fnet6.mode = 'test';
    fnet6.eval({'input', gpuArray(single(Data))});
    Pred6 = gather(fnet6.vars(fnet6.getVarIndex('prob')).value);
    % Fold 7
    fnet6.move('cpu'); fnet7.move('gpu'); fnet7.mode = 'test';
    fnet7.eval({'input', gpuArray(single(Data))});
    Pred7 = gather(fnet7.vars(fnet7.getVarIndex('prob')).value);
    % Fold 8
    fnet7.move('cpu'); fnet8.move('gpu'); fnet8.mode = 'test';
    fnet8.eval({'input', gpuArray(single(Data))});
    Pred8 = gather(fnet8.vars(fnet8.getVarIndex('prob')).value);
    
    Pred = (1/8)*(Pred1 + Pred2 + Pred3 + Pred4 + Pred5 + Pred6 + Pred7 + Pred8);
    [~,PredLab] = max(Pred, [], 3);
    
    toc
    save([ID(1:end-4),'_Pred.mat'], 'PredLab');
end