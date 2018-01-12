function [net, info] = ReLayNet(imdb, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_seg;

	opts.train.extractStatsFn = @extract_stats_segmentation_Mod;
	opts.train.batchSize = 8; % Adjust based on GPU space
	opts.train.numEpochs = 60 ;
	opts.train.continue = true ;
	opts.train.gpus = [2] ;
	opts.train.learningRate = [1e-1*ones(1, 20),  1e-3*ones(1, 20),  1e-4*ones(1, 20)];
	opts.train.weightDecay = 1e-4;
	opts.train.momentum = 0.95;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = false;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;

    
    opts.border = [8 8 8 8]; % tblr
    % Create Sliced Dataset
    imdb = CreateSlicedData(imdb);
    
    % augmenting data - Jitter and Fliplr
    augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.data);
    augLabels = zeros(size(imdb.images.labels) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.labels);
    augData(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data;
    % Mirroring Borders for augData
    augData(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(opts.border(1):-1:1, ...
    :, :, :);
    augData(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augData(:, ...
    opts.border(3):-1:1, :, :) = augData(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augData(:, ...
    end-opts.border(4)+1:end, :, :) = augData(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);

    % Augmenting Labels
    augLabels(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels;
    % Mirroring Borders for augLabels
    augLabels(1:opts.border(1), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels(opts.border(1):-1:1, ...
    :, :, :);
    augLabels(end-opts.border(2)+1:end, ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.labels(end:-1:end-opts.border(2)+1, ...
    :, :, :);
    augLabels(:, ...
    opts.border(3):-1:1, :, :) = augLabels(:, ...
    opts.border(3)+1:2*opts.border(3), :, :);
    augLabels(:, ...
    end-opts.border(4)+1:end, :, :) = augLabels(:, ...
    end-opts.border(4):-1:end-2*opts.border(4)+1, :, :);
    

    imdb.images.augData = augData;
    imdb.images.augLabels = augLabels;
    clear augData augLabels
    
    
	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 3);
	
	% debuging code
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];
    
    % rendering ColorMap for Visualization of 10 classes
    colorMap  = (1/255)*[		    
					    128 128 128
					    128 0 0
					    128 64 128
					    0 0 192
					    64 64 128
					    128 128 0
					    192 192 128
					    64 0 128
					    192 128 128
                       0 255 255
					    ];
	
	opts.train.colorMapGT = [0 0 0; colorMap];
	opts.train.colorMapEst = colorMap;
	NumCls = 10;

	% network definition
	net = dagnn.DagNN() ;
    % Encoder
	net.addLayer('conv1', dagnn.Conv('size', [7 3 1 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 1 1]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});
	net.addLayer('relu1', dagnn.ReLU(), {'bn1'}, {'relu1'}, {});
	net.addLayer('pool1', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1'}, {'pool1', 'pool_indices_1', 'sizes_pre_pool_1', 'sizes_post_pool_1'}, {});

	net.addLayer('conv2', dagnn.Conv('size', [7 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 1 1]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
	net.addLayer('pool2', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2'}, {'pool2', 'pool_indices_2', 'sizes_pre_pool_2', 'sizes_post_pool_2'}, {});
    
    net.addLayer('conv3', dagnn.Conv('size', [7 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 1 1]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('bn3', dagnn.BatchNorm('numChannels', 64), {'conv3'}, {'bn3'}, {'bn3f', 'bn3b', 'bn3m'});
	net.addLayer('relu3', dagnn.ReLU(), {'bn3'}, {'relu3'}, {});
	net.addLayer('pool3', dagnn.PoolingInd('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu3'}, {'pool3', 'pool_indices_3', 'sizes_pre_pool_3', 'sizes_post_pool_3'}, {});
    
    % BottleNeck
    net.addLayer('conv4', dagnn.Conv('size', [7 3 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 1 1]), {'pool3'}, {'conv4'},  {'conv4f'  'conv4b'});
    net.addLayer('bn4', dagnn.BatchNorm('numChannels', 64), {'conv4'}, {'bn4'}, {'bn4f', 'bn4b', 'bn4m'});
	net.addLayer('relu4', dagnn.ReLU(), {'bn4'}, {'relu4'}, {});
    
    % Decoder
    net.addLayer('unpool3x', dagnn.Unpooling(), {'relu4', 'pool_indices_3', 'sizes_pre_pool_3', 'sizes_post_pool_3'}, {'unpool3x'}, {});
    net.addLayer('concat3x', dagnn.Concat('dim',3), {'unpool3x','relu3'}, {'concat3x'});
	net.addLayer('deconv3x', dagnn.Conv('size', [7 3 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [3 3 1 1]), {'concat3x'}, {'deconv3x'},  {'deconv3fx'  'deconv3bx'});
	net.addLayer('bn3x', dagnn.BatchNorm('numChannels', 64), {'deconv3x'}, {'bn3x'}, {'bn3fx', 'bn3bx', 'bn3mx'});
	net.addLayer('relu3x', dagnn.ReLU(), {'bn3x'}, {'relu3x'}, {});
    
    net.addLayer('unpool2x', dagnn.Unpooling(), {'relu3x', 'pool_indices_2', 'sizes_pre_pool_2', 'sizes_post_pool_2'}, {'unpool2x'}, {});
    net.addLayer('concat2x', dagnn.Concat('dim',3), {'unpool2x', 'relu2'}, {'concat2x'});
	net.addLayer('deconv2x', dagnn.Conv('size', [7 3 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [3 3 1 1]), {'concat2x'}, {'deconv2x'},  {'deconv2fx'  'deconv2bx'});
	net.addLayer('bn2x', dagnn.BatchNorm('numChannels', 64), {'deconv2x'}, {'bn2x'}, {'bn2fx', 'bn2bx', 'bn2mx'});
	net.addLayer('relu2x', dagnn.ReLU(), {'bn2x'}, {'relu2x'}, {});
    
    net.addLayer('unpool1x', dagnn.Unpooling(), {'relu2x', 'pool_indices_1', 'sizes_pre_pool_1', 'sizes_post_pool_1'}, {'unpool1x'}, {});
    net.addLayer('concat1x', dagnn.Concat('dim',3), {'unpool1x', 'relu1'}, {'concat1x'});
	net.addLayer('deconv1x', dagnn.Conv('size', [7 3 128 64], 'hasBias', true, 'stride', [1,1], 'pad', [3 3 1 1]), {'concat1x'}, {'deconv1x'},  {'deconv1fx'  'deconv1bx'});
	net.addLayer('bn1x', dagnn.BatchNorm('numChannels', 64), {'deconv1x'}, {'bn1x'}, {'bn1fx', 'bn1bx', 'bn1mx'});
	net.addLayer('relu1x', dagnn.ReLU(), {'bn1x'}, {'relu1x'}, {});


	net.addLayer('classifier', dagnn.Conv('size', [1 1 64 NumCls], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu1x'}, {'classifier'},  {'classf'  'classb'});
	net.addLayer('prob', dagnn.SoftMax(), {'classifier'}, {'prob'}, {});
  	net.addLayer('objective1', dagnn.LossSemantic('weights', 1), {'prob','label'}, 'objective1');
    net.addLayer('objective2', dagnn.LossDice('weights', 1), {'prob','label'}, 'objective2');
	% -- end of the network

	% do the training!
	initNet(net);
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)

if imdb.images.set(batch(1))==1,  % training
  sz0 = size(imdb.images.augData);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
  labels = imdb.images.augLabels(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
    if rand > 0.5, images= fliplr(images) ; labels = fliplr(labels); end % On the fly flip augmentation
else                              % validating / testing
  images = imdb.images.data(:,:,:,batch); 
  labels = imdb.images.labels(:,:,:,batch); 
end

if opts.useGpu > 0
    images = gpuArray(images);
    labels = gpuArray(labels); 
end
inputs = {'input', images, 'label', labels} ;
end

function initNet(net)
	net.initParams();
    
    % He Initialization for New Layers
    for k=1:length(net.layers)
        % is a convolution layer?
        if(strcmp(class(net.layers(k).block), 'dagnn.Conv'))
            f_ind = net.layers(k).paramIndexes(1);
            b_ind = net.layers(k).paramIndexes(2);
            
            [h,w,in,out] = size(net.params(f_ind).value);
            He_gain = 0.7*sqrt(2/(size(net.params(f_ind).value,1)*size(net.params(f_ind).value,2)*size(net.params(f_ind).value,3))); % sqrt(2/fan_in)
            net.params(f_ind).value = He_gain*randn(size(net.params(f_ind).value), 'single');
            net.params(f_ind).learningRate = 1;
            net.params(f_ind).weightDecay = 1;
            
            net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
            net.params(b_ind).learningRate = 0.5;
            net.params(b_ind).weightDecay = 1;
        end
    end
    
end

% Data Slicing during run-time
function imdbNew = CreateSlicedData(imdb)
    sz = size(imdb.images.data);
    sz_ = size(imdb.images.labels);
    numelements = floor(sz(2)/64);
    Rem = mod(sz(2),64);
    Slices = 64*ones(1,numelements);
    SlicedData = mat2cell(imdb.images.data, sz(1), [Slices, Rem], sz(3), sz(4));
    SlicedLabels = mat2cell(imdb.images.labels, sz(1), [Slices, Rem], sz_(3), sz(4));
    SlicedData = SlicedData(1:end-1);
    SlicedLabels = SlicedLabels(1:end-1);
    imdbNew.images.data = cat(4, SlicedData{:});
    imdbNew.images.labels = cat(4, SlicedLabels{:});
    imdbNew.images.set = repmat(imdb.images.set, [numelements,1]);
    imdbNew.images.set = imdbNew.images.set(:);
end
