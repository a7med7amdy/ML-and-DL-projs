# requirements
be sure that all requirements satisfied which are:-

Cython==0.29.4

colorama==0.3.9

numpy==1.15.4

requests==2.21.0

fire==0.1.3

torch==0.4.1

matplotlib==2.2.3

numba==0.39.0

scipy==1.1.0

h5py==2.8.0

pandas==0.23.4

tqdm==4.29.1

tensorboardX==1.6

opencv_python==3.4.3.18

torch==0.4.1

torchvision==0.2.1

tkinter

it's better to satisfy the later versions of those.

run make.sh after that

you can find them in maskSiam object tracking/SiamMask-master (copy)

download the models 
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
then add them in experiments/siammask_sharp

# the apps
when you run thethe main.py, it will ask you to enter the path to the video. then it will ask you

if you want the interactive app, enter 1 and if you want the automatic app, enter 2 (or any other button).

# interactive app
 the interactive app is an app that will show you the first frame of the video and waits you to enter the boundind boxes
 using the mouse (draw it around the object need to be tracked).

 when you finish, press ESC and it will start tracking objects.

 # automatic app

 if you choose the automatic app, it will ask you to enter a path to the csv file.

 it assumes that you will enter a csv file with looks like:-


        name	    x1	y1	x2	y2

0	F_000000000.jpg	411	155	431	175	

1	F_000000028.jpg	284	106	300	122	

2	F_000000037.jpg	281	120	301	142	

3	F_000000037.jpg	366	150	391	176	

name is the name of the frame (its number), then the topleft and bottomright corners of the object you want to track at this frame

then it will track the objects

# important notes in automatic app

if you want to track an object, please enter its bounding box in the csv file once. not with each frame it appears on.

the more objects to track, the slower is the app.

to exit tracking mode at any time, press "ESC"
 
