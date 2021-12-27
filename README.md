# Expression-Identification

## Introduction

This project aims to classify the expression on a person's face into one of seven categories, using deep convolutional neural networks. This project is an implementation of research paper provided in the folder. The model is trained on the FER-2013 dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.



## Dependencies

* Python 3 
* OpenCV (https://opencv.org/)
* Tensorflow (https://www.tensorflow.org/)





## Basic Usage


The Project is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First enter the folder

```
cd Expression-Identification

```



* Download the FER-2013 dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and unzip it inside the `src` folder. This will create the folder `data`.



* If you want to train this model, use:

```

cd src

python emotions.py --mode train

```


* If you want to view the predictions then run:  

```
cd src

python emotions.py --mode display

```


* The folder structure is of the form:  
  
   src:
  
   * data (folder)
  
   * `emotions.py` (file)
  
   * `haarcascade_frontalface_default.xml` (file)
  
   * `model.h5` (file)



* This implementation by default detects emotions on all faces in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 63.2% in 50 epochs.



![Accuracy plot](imgs/accuracy.png)


## Algorithm



* First, the **haar cascade** method is used to detect faces in each frame of the webcam feed.



* The region of image containing the face is resized to **48x48** and is passed as input to the CNN.



* The network outputs a list of **softmax scores** for the seven classes of emotions.



* The emotion with maximum score is displayed on the screen.



## Example Output



![Mutiface](imgs/multiface.png)



## References



* "Challenges in Representation Learning: A report on three machine learning contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
   Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,  
   X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
   M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
   Y. Bengio. arXiv 2013.
* https://github.com/atulapra/Emotion-detection
