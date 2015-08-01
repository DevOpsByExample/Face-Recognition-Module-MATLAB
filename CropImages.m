clear all;
close all;

name = '1.jpg';
s3 = '.pgm';
save = strcat(s1,s3);
        
 A = imread(name);
 FaceDetector = vision.CascadeObjectDetector('MergeThreshold',8);

 BBOX = step(FaceDetector, A);

 B = insertObjectAnnotation(A, 'rectangle', BBOX, 'Face');


  Face=imcrop(B,BBOX);
  I = rgb2gray(Face);
        
  J = imresize(I,[112 92]);
        
  imwrite(J,save);
  figure,imshow(J);

