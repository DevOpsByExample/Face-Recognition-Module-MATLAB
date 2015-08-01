
%%

facedatabase = imageSet('data','recursive');




%%

figure;
montage(facedatabase(1).ImageLocation);
title('Image of Single faces');

% 
% persontoquery =1;
% 
% galleryImage = read(facedatabase(persontoquery),1);
% figure;
% 
% 
% 
% for i=1:size(facedatabase,2)
%     imageset = read(facedatabase(i),1);
% end
% 
% montage(imageset);



%%


[training,test] = partition(facedatabase,[0.8 0.2]);



%%



person = 1;
[hogFeature, visualization]= ...
    extractHOGFeatures(read(training(person),1));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('Hog feature');





%%


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



%%


faceClassifier = fitcecoc(trainingFeatures,trainingLabel);


            
%%


person =1;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(queryImage);
personLabel = predict(faceClassifier,queryFeatures);


booleanIndex = strcmp(personLabel,personIndex);
integerIndex = find(booleanIndex);
subplot(1,2,1);imshow(queryImage);title('Query Image');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched');






%%


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


%%
