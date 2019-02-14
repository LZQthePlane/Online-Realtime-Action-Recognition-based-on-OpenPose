# Online-Action-Recognition-based-on-Openpose
A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints. Using openpose as the online pose estimator.   
(The code comments are partly descibed in chinese)


------
## Introduction
*The **pipline** of this work is:*   
 - Realtime pose estimation by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);   
 - Online human tracking for multi-people scenario by [DeepSort algorithm](https://github.com/nwojke/deep_sortv);   
 - Action recognition with DNN for each person based on single framewise joints detected from Openpose.


------
## Dependencies
 - Opencv > 3.4.1   
 - sklearn
 - tensorflow & keras
 - numpy & scipy 
 - pathlib
 
 
------
## Usage
 - Download the openpose VGG tf-model with command line `./download.sh`(/Pose/graph_models/VGG_origin) or fork [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A#list/path=%2Fsharelink1864347102-902260820936546%2Fopenpose%2Fopenpose%20graph%20model%20coco&parentPath=%2Fsharelink1864347102-902260820936546), and place it under the corresponding folder; 
 - `python main.py`, it will **start the webcam**. 
 (you can choose to test video with command `python main.py --video=test.mp4`, however I just tested the webcam mode)   
 - By the way, you can choose different openpose pretrained model in script.    
 **graph_opt.pb**: training with the VGG net, as same as the CMU providing caffemodel, more accurate but slower, **graph_opt_mobile.pb**:  training with the Mobilenet, much smaller than the origin VGG, faster but less accurate.   
 **However, Please attention that the Action Dataset in this repo is collected with the** ***VGG model.***


------
## Test result
<p align="center">
    <img src="https://github.com/LZQthePlane/Online-Action-Recognition-based-on-Openpose/blob/master/test_out/webcam_test_out.gif", width="840">
 

-------
## Note
 - Action recognition in this work is framewise based, so it's technically "**Pose recognition**" to be exactly;   
 - Action is actually a dynamic motion which consists of sequential static poses, therefore classifying framewisely is not a good solution.
 - Considering of using ***RNN(LSTM) model*** to classify actions with dynamic sequential joints data is the next step to improve this project.


------
## Reference
Thanks for the following awesome works:    
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation),   
 - [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3),    
 - [Real-Time-Action-Recognition](https://github.com/TianzhongSong/Real-Time-Action-Recognition)
