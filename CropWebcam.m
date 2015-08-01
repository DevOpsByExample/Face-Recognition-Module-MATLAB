clear all;
close all;

cam = webcam();
preview(cam);

for idx = 1:10
   
   s1 = int2str(idx);
   s2 = '.jpg';
   
   s3 = '.pgm';
    

   s = strcat(s1,s2);
   save = strcat(s1,s3);
   
   rgbImage = snapshot(cam);
   imwrite(rgbImage,s);
   A = imread(s);
   
   FaceDetector = vision.CascadeObjectDetector();
   BBOX = step(FaceDetector, A);
   
   Face=imcrop(A,BBOX);
  
   grayImage = rgb2gray(Face); 
   J = imresize(grayImage,[112 92]);
   imwrite(J,save);

   imshow(J);
   hold on;
   
end

clear cam;
