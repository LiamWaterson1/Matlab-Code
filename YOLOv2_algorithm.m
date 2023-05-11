pretrainedURL = 'https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/yolov2/tiny_yolov2.tar';
pretrainedNetTar = 'yolov2Tiny.tar';

if ~exist(pretrainedNetTar,'file')
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetTar,pretrainedURL);
end

onnxfiles = untar(pretrainedNetTar);
pretrainedNet = fullfile('tiny_yolov2','Model.onnx');
lgraph = importONNXLayers(pretrainedNet,'ImportWeights',true);
lgraph = removeLayers(lgraph,'RegressionLayer_grid');
onnxAnchors = [1.08,1.19; 3.42,4.41; 6.63,11.38; 9.42,5.11; 16.62,10.52];
tic;
inputSize = lgraph.Layers(1,1).InputSize(1:2);
lastActivationSize = [13,13];
upScaleFactor = inputSize./lastActivationSize;
anchorBoxesTmp = upScaleFactor.* onnxAnchors;
anchorBoxes = [anchorBoxesTmp(:,2),anchorBoxesTmp(:,1)];
weights = lgraph.Layers(end,1).Weights;
bias = lgraph.Layers(end,1).Bias;
layerName = lgraph.Layers(end,1).Name;

numAnchorBoxes = size(onnxAnchors,1);
[modWeights,modBias] = rearrangeONNXWeights(weights,bias,numAnchorBoxes);
filterSize = size(modWeights,[1 2]);
numFilters = size(modWeights,4);
modConvolution8 = convolution2dLayer(filterSize,numFilters,...
    'Name',layerName,'Bias',modBias,'Weights',modWeights);
lgraph = replaceLayer(lgraph,'convolution8',modConvolution8);

classNames = tinyYOLOv2Classes;

layersToAdd = [
    yolov2TransformLayer(numAnchorBoxes,'Name','yolov2Transform');
    yolov2OutputLayer(anchorBoxes,'Classes',classNames,'Name','yolov2Output');
];

lgraph = addLayers(lgraph, layersToAdd);
lgraph = connectLayers(lgraph,layerName,'yolov2Transform');
yoloScaleLayerIdx = find(...
    arrayfun( @(x)isa(x,'nnet.onnx.layer.ElementwiseAffineLayer'), ...
    lgraph.Layers));

if ~isempty(yoloScaleLayerIdx)
    for i = 1:size(yoloScaleLayerIdx,1)
        layerNames {i} = lgraph.Layers(yoloScaleLayerIdx(i,1),1).Name;
    end
    lgraph = removeLayers(lgraph,layerNames);
    lgraph = connectLayers(lgraph,'image','convolution');
end
load("matlab.mat");
net = assembleNetwork(lgraph);
yolov2Detector = yolov2ObjectDetector(net);

[filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp','Image files'});
path=[pathname,filename];
I = imread(path);

% Convert image to BGR format.
Ibgr = cat(3,I(:,:,3),I(:,:,2),I(:,:,1));
[bboxes, scores, labels] = detect(yolov2Detector, Ibgr);

% calculate recall, precision, and f1 score
[~,~,numGtBoxes] = size(ground_truth);
if isempty(bboxes)
    numDetBoxes = 0;
else
    numDetBoxes = size(bboxes, 1);
end
[precision, recall, f1_score] = evaluate_detection(bboxes, ground_truth);
time = toc;
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 score: %.2f\n', f1_score);
fprintf('Time: %.5f\n', time);
function [modWeights,modBias] = rearrangeONNXWeights(weights,bias,numAnchorBoxes)
%rearrangeONNXWeights rearranges the weights and biases of an imported YOLO
%v2 network as required by yolov2ObjectDetector. numAnchorBoxes is a scalar
%value containing the number of anchors that are used to reorder the weights and
%biases. This function performs the following operations:
%   * Extract the weights and biases related to IoU, boxes, and classes.
%   * Reorder the extracted weights and biases as expected by yolov2ObjectDetector.
%   * Combine and reshape them back to the original dimensions.

weightsSize = size(weights);
biasSize = size(bias);
sizeOfPredictions = biasSize(3)/numAnchorBoxes;

% Reshape the weights with regard to the size of the predictions and anchors.
reshapedWeights = reshape(weights,prod(weightsSize(1:3)),sizeOfPredictions,numAnchorBoxes);

% Extract the weights related to IoU, boxes, and classes.
weightsIou = reshapedWeights(:,5,:);
weightsBoxes = reshapedWeights(:,1:4,:);
weightsClasses = reshapedWeights(:,6:end,:);

% Combine the weights of the extracted parameters as required by
% yolov2ObjectDetector.
reorderedWeights = cat(2,weightsIou,weightsBoxes,weightsClasses);
permutedWeights = permute(reorderedWeights,[1 3 2]);

% Reshape the new weights to the original size.
modWeights = reshape(permutedWeights,weightsSize);

% Reshape the biases with regared to the size of the predictions and anchors.
reshapedBias = reshape(bias,sizeOfPredictions,numAnchorBoxes);

% Extract the biases related to IoU, boxes, and classes.
biasIou = reshapedBias(5,:);
biasBoxes = reshapedBias(1:4,:);
biasClasses = reshapedBias(6:end,:);

% Combine the biases of the extracted parameters as required by yolov2ObjectDetector.
reorderedBias = cat(1,biasIou,biasBoxes,biasClasses);
permutedBias = permute(reorderedBias,[2 1]);

% Reshape the new biases to the original size.
modBias = reshape(permutedBias,biasSize);
end

function [precision, recall, f1_score] = evaluate_detection(bboxes, ground_truth)
    num_detections = size(bboxes, 1);
    num_ground_truth = size(ground_truth, 1);
    t_p = 0;
    f_p = 0;
    f_n = 0;
    
    % Match detections and work out truths
    for i = 1:num_detections
        matched = false;
        for j = 1:num_ground_truth
            overlap = bboxOverlapRatio(bboxes(i,:), ground_truth(j,:));
            if overlap > 0.5 
                t_p = t_p + 1;
                matched = true;
                break;
            end
        end
        if ~matched
            f_p = f_p + 1;
        end
    end  
    % count miss detections
    for i = 1:num_ground_truth
        matched = false;
        for j = 1:num_detections
            overlap = bboxOverlapRatio(bboxes(j,:), ground_truth(i,:));
            if overlap > 0.5 
                matched = true;
                break;
            end
        end
        if ~matched
            f_n = f_n + 1;
        end
    end    
    % calculate the values for precision, recall and f1 score
    precision = t_p / (t_p + f_p);
    recall = t_p / (t_p + f_n);
    f1_score = 2 * (precision * recall) / (precision + recall);
end
function classes = tinyYOLOv2Classes()
% Return the class names corresponding to the pretrained ONNX tiny YOLO v2
% network.
%
% The tiny YOLO v2 network is pretrained on the Pascal VOC data set,
% which contains images from 20 different classes.

classes = [ ...
    " aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",...
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",...
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
end