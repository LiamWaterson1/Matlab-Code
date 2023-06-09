clear; % clear the workspace
faceDetector = vision.CascadeObjectDetector(); 
% user selects image
[filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp','Image files'});
path=[pathname,filename];
% Load the image
FaceImage = imread(path);
tic;
% Detect faces and return their locations
bboxes = step(faceDetector,FaceImage);
load('matlab.mat'); %load the ground truth file
% find and display the values
[precision, recall, f1_score] = evaluate_detection(bboxes, ground_truth);
time = toc; %stop measuring time
% Print results
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1 score: %.2f\n', f1_score);
fprintf('Time: %.5f\n', time);
% Draw boxes on the detected faces
FaceOut = insertShape(FaceImage,'Rectangle',bboxes,'Color','red');
% Display the output
figure, imshow(FaceOut), title('Detected faces');
% Clean up
release(faceDetector);
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