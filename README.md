# Face detection base on ResnetSSD
Project of Face detection using Resnet and SSD, apply in image and webcam.   
(The code comments are descibed in chinese)

------
## ***Folder Intro***
### —test_out
The result of my test in webcam.

### —Resnet_SSD_deploy/Res10_300x300_SSD_iter_140000.caffemodel
Files save the pre-trained SSD-Resnet caffe model.    
   - .prototxt file specifies the architecture of the neural network – how the different layers are arranged etc.
   - .caffemodel file stores the weights of the trained model.    
   
OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network (*unlike other OpenCV SSDs that you may have seen which typically use MobileNet as the base network).*

------
## Result
![my test](https://github.com/LZQthePlane/Face-detection-base-on-ResnetSSD/blob/master/test_out/example.gif) 
