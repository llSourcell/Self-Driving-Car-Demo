# Self Driving Car Demo

Overview
============
A project that trains a virtual car to how to move an object around a screen (drive itself) without running into obstacles using a type of reinforcement learning called Q-Learning. More information can be found on the writeup about this project in [part 1](https://medium.com/@harvitronix/using-reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-6e782cc7d4c6), [part 2](https://medium.com/@harvitronix/reinforcement-learning-in-python-to-teach-a-virtual-car-to-avoid-obstacles-part-2-93e614fcd238#.vbakopk4o), and [part 3](https://medium.com/@harvitronix/reinforcement-learning-in-python-to-teach-an-rc-car-to-avoid-obstacles-part-3-a1d063ac962f). This the code for 'Build an Self Driving Car in 5 Min' on [Youtube](https://youtu.be/hBedCdzCoWM)

Dependencies
============

* Numpy (http://www.numpy.org/)
* Pygame. I used these instructions: http://askubuntu.com/questions/401342/how-to-download-pygame-in-python3-3 but with ```pip3 install hg+http://bitbucket.org/pygame/pygame``` after I installed the dependencies
* Pymunk. Use Pymunk 4. Using version 5 will not work as it was a major refactor. V4 is [here](https://github.com/viblo/pymunk/releases/tag/pymunk-4.0.0)
* Keras ```pip3 install keras```
* Theanos ```pip3 install git+git://github.com/Theano/Theano.git --upgrade --no-deps```
* h5py ```pip3 install h5py```

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========

1. Update pymunk to python3 by CDing into its directory and running  ```2to3 -w *.py```

2. First, you need to train a model. This will save weights to the `saved-models` folder. *You may need to create this folder before running*. You can train the model by running `python3 learning.py` It can take anywhere from an hour to 36 hours to train a model, depending on the complexity of the network and the size of your sample. However, it will spit out weights every 25,000 frames, so you can move on to the next step in much less time.

3. Edit the `nn.py` file to change the path name for the model you want to load. Sorry about this, I know it should be a command line argument. Then, watch the car drive itself around the obstacles! Run `python3 playing.py`

4. Once you have a bunch of CSV files created via the learning, you can convert those into graphs by running: `python3 plotting.py`

This will also spit out a bunch of loss and distance averages at the different parameters. That's it! 

Credits
===========
Credit for the vast majority of code here goes to [Harvitronix](https://github.com/harvitronix/reinforcement-learning-car). I've merely created a wrapper around all of the important functions to get people started. Below are a few sources he cited. 

- Playing Atari with Deep Reinforcement Learning - http://arxiv.org/pdf/1312.5602.pdf
- Deep learning to play Atari games: https://github.com/spragunr/deep_q_rl
- Another deep learning project for video games: https://github.com/asrivat1/DeepLearningVideoGames
- A great tutorial on reinforcement learning that a lot of my project is based on: http://outlace.com/Reinforcement-Learning-Part-3/
