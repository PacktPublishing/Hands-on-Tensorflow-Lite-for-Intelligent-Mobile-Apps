# Hands-on-Tensorflow-Lite-for-Intelligent-Mobile-Apps
Hands on Tensorflow Lite for Intelligent Mobile Apps, published by Packt

Notes by the author:

# Repository
This repository is divided into 5 sections (excluding the first one because it was theoretical) and a "demo" code from Tensorflow which might be recommended to run (see below "On using the camera of Android emulator").

Each of these folders may contain:

* The code used during the tutorial, whose name typically starts by the number of the section and video.
* Logs and results. Sometimes there is a "log" folder typically used by Tensorboard and logRes files where the accuracy of the different configurations is shown.
* tmp folder, which contains the graph (frozen and unfrozen), weights, and TF Lite converted model.
* Folders containing the apps developed in iOS and Android shown in the tutorials.
* A dataset folder (section 3 to 6)

# On compiling with bazel
As I've mention in the videos, I had some problems on Ubuntu compiling freeze_graph and toco to freeze graphs and transform frozen graphs into tensorflow lite format respectively. On MacOS Sierra I had no issues though.

In order to compile both files, I simply run this line:

A) Graph freezer:
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow/python/tools:freeze_graph

B) TF Lite conversor:
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 tensorflow/contrib/lite/toco:toco

# On using the camera of the Android emulator
As you can see in the videos, I had problems when displaying the camera, but it actually works when I take pictures. This issue might happen with the API level I used (23) since when I develop using API 26 the camera works. I use level 23 because this way it's compatible, but I encourage you to try Android level 24.

If for some reason the camera makes the app crashes, you can try to run first the "demo" application to see if you can execute it.
