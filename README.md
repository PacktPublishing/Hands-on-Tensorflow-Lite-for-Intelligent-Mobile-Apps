# Hands-on TensorFlow Lite for Intelligent Mobile Apps [Video]
This is the code repository for [Hands-on TensorFlow Lite for Intelligent Mobile Apps [Video]](https://www.packtpub.com/application-development/hands-tensorflow-lite-intelligent-mobile-apps-video?utm_source=github&utm_medium=repository&utm_campaign=9781788990677), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the video course from start to finish.
## About the Video Course
This complete guide will teach you how to build and deploy Machine Learning models on your mobile device with TensorFlow Lite. You will understand the core architecture of TensorFlow Lite and the inbuilt models that have been optimized for mobiles. 
You will learn to implement smart data-intensive behavior, fast, predictive algorithms, and efficient networking capabilities with TensorFlow Lite. You will master the TensorFlow Lite Converter, which converts models to the TensorFlow Lite file format. This course will teach you how to solve real-life problems related to Artificial Intelligence—such as image, text, and voice recognition—by developing models in TensorFlow to make your applications really smart. You will understand what Machine Learning can do for you and your mobile applications in the most efficient way. With the capabilities of TensorFlow Lite you will learn to improve the performance of your mobile application and make it smart.
By the end of the course, you will have learned to implement AI in your mobile applications with TensorFlow.

<H2>What You Will Learn</H2>
<DIV class=book-info-will-learn-text>
<UL>
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Explore the current state of Machine Learning and Artificial Intelligence.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Develop the understanding to build AI systems using different machine learning models.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Optimize machine learning models for better performance and accuracy.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Understand different deep learning models for computer vision</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Explore generative models and how they generate information from random noise.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Code the most trending AI algorithms that outperform humans in video games.</SPAN> </LI></UL></DIV>

## Instructions and Navigation
### Assumed Knowledge
To fully benefit from the coverage included in this course, you will need:<br/>
No knowledge of Deep Learning or TensorFlow is required since both theoretical and practical aspects will be introduced, but basic mobile application development knowledge is assumed since the focus of the course is on solving Artificial Intelligence-related problems.
### Technical Requirements
This course has the following software requirements:<br/>
For Python either 2.7.X or 3.X
For Tensorflow, I used 1.4.1 but probably the last current version would work.

## Related Products
* [TensorFlow 1.X Recipes for Supervised and Unsupervised Learning [Video]](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-1x-recipes-supervised-and-unsupervised-learning-video?utm_source=github&utm_medium=repository&utm_campaign=9781788398756)

* [TensorFlow for Machine Learning Solutions [Video]](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-solutions-video?utm_source=github&utm_medium=repository&utm_campaign=9781789136272)

* [Hands-on Artificial Intelligence with TensorFlow [Video]](https://www.packtpub.com/big-data-and-business-intelligence/hands-artificial-intelligence-tensorflow-video?utm_source=github&utm_medium=repository&utm_campaign=9781789135091)

# Repository
-This repository is divided into 5 sections (excluding the first one because it was theoretical) and a "demo" code from Tensorflow which might be recommended to run (see below "On using the camera of Android emulator").
-
-Each of these folders may contain:
-
-* The code used during the tutorial, whose name typically starts by the number of the section and video.
-* Logs and results. Sometimes there is a "log" folder typically used by Tensorboard and logRes files where the accuracy of the different configurations is shown.
-* tmp folder, which contains the graph (frozen and unfrozen), weights, and TF Lite converted model.
-* Folders containing the apps developed in iOS and Android shown in the tutorials.
-* A dataset folder (section 3 to 6)
-* freeze_graph file containing the commands used to freeze the graph and convert the model into Tensorflow Lite.
-
-# On compiling with bazel
-As I've mention in the videos, I had some problems on Ubuntu compiling freeze_graph and toco to freeze graphs and transform frozen graphs into tensorflow lite format respectively. On MacOS Sierra I had no issues though.
-
-In order to compile both files, I simply run this line:
-
-A) Graph freezer:
-bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow/python/tools:freeze_graph
-
-B) TF Lite conversor:
-bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow/contrib/lite/toco:toco
-
-# On using the camera of the Android emulator
-As you can see in the videos, I had problems when displaying the camera, but it actually works when I take pictures. This issue might happen with the API level I used (23) since when I develop using API 26 the camera works. I use level 23 because this way it's compatible, but I encourage you to try Android level 24.
-
-If for some reason the camera makes the app crashes, you can try to run first the "demo" application to see if you can execute it.

<H2>What You Will Learn</H2>
<DIV class=book-info-will-learn-text>
<UL>
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Explore the current state of Machine Learning and Artificial Intelligence.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Develop the understanding to build AI systems using different machine learning models.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Optimize machine learning models for better performance and accuracy.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Understand different deep learning models for computer vision</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Explore generative models and how they generate information from random noise.</SPAN> 
<LI><SPAN id=what_you_will_learn_c class=sugar_field>Code the most trending AI algorithms that outperform humans in video games.</SPAN> </LI></UL></DIV>


