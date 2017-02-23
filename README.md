# Raspberry-Kinect-Opencv
Spheres Recognition Color-Depth

Hi there! This is my first project,
First version

This is an artifitial vision system for robotics application, developed with Python and OpenCV, acquiring the images with a Kinect sensor and processing them with a Raspberry Pi 3 Model B.

The system recognizes spheres and their position on coordinates x, y, in cm respect the position of the Kinect Sensor.

Recognizes only two colors (orange and green).
The Kinect sensor used is the xBox360 Kinect Sensor - 1414

1. Install Raspbian on your Raspberry Pi - https://www.raspberrypi.org/downloads/
2. Install the OpenCV library for Python - http://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/
3. Install the Numpy library for python - pip install numpy
4. Install libfreenect to be able to use Kinect sensor - Nice tutorial -> https://naman5.wordpress.com/2014/06/24/experimenting-with-kinect-using-opencv-python-and-open-kinect-libfreenect/ AND For more information about the OpenKinect community -> https://openkinect.org/wiki/Main_Page

For finding spheres, this system uses the HougCircle method. If the green and orange colors are not well filtered, you can change the ranges of the colors desired, it is used the Lab colorspace.
  
Thanks,
Snigf12
