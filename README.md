# 1 shot learnig for MNIST data
cf. https://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf

* command 
<br>main.py

<h3> training </h3>

* randomly choose 5 images for each of "0" to "7" class
* make all the 80 pairs of images from same class and random 80 pairs of images from different class
* the label is 1 for same_pair,0 for different_pair
<h3> one-shot learning </h3>

* randomly choose only one "8" image and only one "9" image as sample
* classify "8" and "9" images by the trained model and the samples for "8" and "9"
<h3> score one-shot leaning </h3>
* repeat one-shot learning for 10 times and calculate the average accuracy