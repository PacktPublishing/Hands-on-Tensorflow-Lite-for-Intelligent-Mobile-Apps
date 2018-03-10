# Hands-on-Tensorflow-Lite-for-Intelligent-Mobile-Apps
Hands on Tensorflow Lite for Intelligent Mobile Apps, published by Packt

Notes by the author:

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
