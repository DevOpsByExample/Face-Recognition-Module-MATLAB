% clear all;
% close all;
% clc;

while (1==1)
    choice=menu('Face Recognition',...
                'Generate Database',...
                'Train system',...
                'Feature Extraction',...
                'Generate Classifier',...
                'Test',...
                'Open Image From System',...
                'Exit');
    
    %% Generating Database " loading dataset into matlab"
        if (choice ==1)
         
            choice2 = questdlg('Generate a new database?', ...
                               'Warning...',...
                               'Yes', ...
                               'No','No');            
            switch choice2
                case 'Yes'
                    pause(0.1);
                    facedatabase = imageSet('data','recursive');        
                case 'No'
            end
              
    end
    
    %%Establishing Training set and test set by dividing the dataset in 8:2 
    if (choice == 2)
       if (~exist('facedatabase','var'))
            warndlg('Please generate database first!\n');
       else
             [training,test] = partition(facedatabase,[0.8 0.2]);
       end
    end
    
    %%Feature Extraction 
    if (choice == 3)
         if (~exist('facedatabase','var'))
              warndlg('Please generate database first!'); 
              %fprintf('Please generate database first!\n');
         elseif(~exist('test','var') || ~exist('training','var'))
              warndlg('Training the system first!\n');
         else
             trainingFeatures = zeros(size(training,2)*training(1).Count,4680);
             featureCount = 1;
            for i=1:size(training,2)
                    for j= 1:training(i).Count
                        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),1));
                        trainingLabel{featureCount} = training(i).Description ;
                        featureCount = featureCount +1;
                    end
                    personIndex{i}= training(i).Description;
            end 
         end
    end
    
    %%Generating Classfier using ecoc machine learning algorithm 
    if (choice == 4)
         if (~exist('facedatabase','var'))
              warndlg('Please generate database first!'); 
         elseif(~exist('test','var') || ~exist('training','var'))
              warndlg('Training the system first!\n');
         elseif(~exist('trainingFeatures','var'))
             warndlg('Extract Features before proceding to classifer!'); 
         else
             faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
         end
    end
    
    %% Prediction "Recognition"
    if (choice == 5)
         if (~exist('facedatabase','var'))
              warndlg('Please generate database first!'); 
         elseif(~exist('test','var') || ~exist('training','var'))
              warndlg('Training the system first!\n');
         elseif(~exist('trainingFeatures','var'))
             warndlg('Extract Features before proceding to classifer!');
         elseif(~exist('faceClassifier','var'));
             warndlg('Run Classifier First');
         else
                figure;
                figureNum =1;
                for person = 1:5 
                     for j =1:2
                        queryImage = read(test(person),j);
                        queryFeatures = extractHOGFeatures(queryImage);
                        personLabel = predict(faceClassifier,queryFeatures);
                        booleanIndex = strcmp(personLabel,personIndex);
                        integerIndex = find(booleanIndex);
                        subplot(5,4,figureNum);imshow(imresize(queryImage,3));title('QueryImage');
                        subplot(5,4,figureNum+1);imshow(read(training(integerIndex),1)); title('Match');
                        figureNum = figureNum+2;
                     end
                end
         end
    end    
    
    %% Reading image from a path and recognising the person from the dataset
    if (choice == 6)
         if (~exist('facedatabase','var'))
              warndlg('Please generate database first!'); 
         elseif(~exist('test','var') || ~exist('training','var'))
              warndlg('Training the system first!\n');
         elseif(~exist('trainingFeatures','var'))
             warndlg('Extract Features before proceding to classifer!');
         elseif(~exist('faceClassifier','var'));
             warndlg('Run Classifier First');
        else            
            pause(0.1);            
            [file_name file_path] = uigetfile ({'*.pgm';'*.jpg';'*.png'});
            if file_path ~= 0
                filename = [file_path,file_name]; 
                queryImage = imread(filename);
                queryFeatures = extractHOGFeatures(queryImage);
                personLabel = predict(faceClassifier,queryFeatures);
                booleanIndex = strcmp(personLabel,personIndex);
                integerIndex = find(booleanIndex);
                subplot(1,2,1);imshow(queryImage);title('Query Image');
                subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched');
            end
        end
    end
    
    if (choice == 7)
        clear choice choice2
        return;
    end    
end
