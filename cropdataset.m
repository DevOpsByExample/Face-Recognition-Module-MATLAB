
%%To Crop the Faces Automatically from a Dataset folder
clear all;
close all;

for i = 1:10   
   s1 = int2str(i);
   s2 = '.jpg';
   s3 = '.pgm';
   s4 = 'dataset/'; 
   s = strcat(s4,s1,s2);
   save = strcat(s1,s3);
   A = imread(s);
   
   FaceDetector = vision.CascadeObjectDetector();
   BBOX = step(FaceDetector, A);
   
   Face=imcrop(A,BBOX);
  
   grayImage = rgb2gray(Face); 
   J = imresize(grayImage,[112 92]);
   imwrite(J,save);
 
   imshow(J); 
end