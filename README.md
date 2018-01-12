# ReLayNet

Code and Trained Models

-----------------
If you use this code, please cite:

Abhijit Guha Roy, Sailesh Conjeti, Sri Phani Krishna Karri, Debdoot Sheet, Amin Katouzian, Christian Wachinger, and Nassir Navab, 
"ReLayNet: retinal layer and fluid segmentation of macular optical coherence tomography using fully convolutional networks," 
Biomed. Opt. Express 8, 3627-3642 (2017) 

Coded by: A. Guha Roy and S. Conjeti

------------------


Usage: 

1. Download MatConvNet and Compile it (Follow: http://www.vlfeat.org/matconvnet/install/)

2. Unzip ReLayNet folder. Copy files:

Copy /layers/matlab files/ ---> <MatConvNet_HomeFolder>/matlab/

Copy /layers/dagnn wrappers/ ---> <MatConvNet_HomeFolder>/matlab/+dagnn/

3. Copy Rest of the files in another home Folder

4. Create an experiment Folder ex: 'Exp01_ReLayNet_ChoroidSegmentation'

5. Create Imdb of the dataset (To know how to create imdb, refer: http://germanros.net/online-courses/hands-on-dl/)
It is basically a structure: imdb.images.data is a 4D tensor as [height, width, channel, NumberOfData]
			     imdb.images.labels is a 4D tensor as [height, width, 2, NumberOfData] ---> 1st Channel is class (1,2,... etc), 2nd channel is Instance Weights (Refer the paper)
			     imdb.images.set is [1,NumberOfData] vector with entries 1 or 3 indicating which data is for training and validation respectively.

6. RunTraining: 

[net, info] = ReLayNet(imdb, inpt); where initialize, inpt.expDir = 'Exp01_ReLayNet_ChoroidSegmentation'

7. In the code check the hyper parameters like learning rate, number of class, epochs etc

-----------------------------
