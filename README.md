# Tensorflow Vs Keras? — Comparison by building a model for image classification.

![](https://cdn-images-1.medium.com/max/2000/1*99EOCVmmez8FK6pdQCNUsQ.jpeg)

Yes , as the title says , it has been very usual talk among data-scientists
(even you!) where a few say , TensorFlow is better and some say Keras is way
good! Let’s see how this thing actually works out in practice in the case of
image classification.

Before that let’s introduce these two terms Keras and Tensorflow and help you
build a powerful image classifier within 10 min!

### **Tensorflow:**

Tensorflow is the most used library to develop models in deep learning. It has
been the best ever library which has been completely opted by many geeks in
their daily experiments . Could you imagine if I say that Google has put Tensor
Processing Units (TPU) just to deal with tensors ? Yes, they have. They have put
a separate class of instances called TPU which has the most power driven
computational power to handle the deep learning models of tensorflow.

### **Time to BUILD IT!**

I’ll now help you create a powerful image classifier using tensorflow. Wait!
what is a classifier? It’s just a simple question you throw to your tensorflow
code asking whether the given image is a rose or a tulip. So , first things
first.Let us install tensorflow on the machine. There are two versions proivided
by the official documentation i.e., CPU and the GPU version. For CPU version :

    pip install tensorflow

And please note , I am writing this blog after experimenting on a GPU and NOT
CPU. GPU installation is neatly given
[here](https://www.tensorflow.org/install/).

![](https://cdn-images-1.medium.com/max/1600/1*dr3pZsLJg28gKwq0MXp1Mg.png)

Now , let us take the Google’s Tensorflow for poets experiment to train a model.
This repository of Google has amazing scripts for easy experiments on images. It
is very much concise and sufficient for our purpose. Remember the word
**powerful** which I used before? Yes , that word comes into action when we use
something called **transfer learning. **Transfer learning is a powerful way of
using the pre-trained models which have been trained for days or might be weeks
and then changing the final layer to adjust to our own set of classes.

Inception V3 is a very good model which has been ranked 2nd in [2015 ImageNet
Challenge](http://image-net.org/challenges/LSVRC/2015/results) for image
classification. It has been mentioned as the best network for transfer learning
for datasets with less number of images per class.

![](https://cdn-images-1.medium.com/max/2000/0*9YouBEcHMPaeLJ1W.png)
<span class="figcaption_hack">Inception V3</span>

Now clone the git repository:



Now , you get to choose your images . All you have to do is put the dataset
folder in the below fashion.

     — Dataset folder -
           class1/
               — image1
               — image2
           class2/
               — image1
               — image2

![](https://cdn-images-1.medium.com/max/1600/1*I_lY47Dg6Px_WMTfu3OKhg.png)
<span class="figcaption_hack">FLOWER DATA</span>

It should look something like above (Ignore the image.py). I have got the above
**flower_photos **folder by:


#### Creating the Dataset

You can use whatever images you’d like. The more the better (aim for a few
thousand). Separate them by categories as done above, and make sure they are in
a folder called `tf_files.`

You can download pre-exiting datasets of various use cases like cancer detection
to characters in Game of Thrones. Here is various [image classification
datasets.](https://dataturks.com/projects/Trending?type=IMAGE_CLASSIFICATION)

Or if you have your unique use case, you can create your very own dataset for
it. You can download images from the web and to make a big dataset in no time,
use an annotation tool like [Dataturks](https://dataturks.com/), where you
upload the images and tag images manually in a ziffy. Better yet, the output
from Dataturks can be easily used to building the tf_files.

![](https://cdn-images-1.medium.com/max/1600/1*IMKNwK4Bhqsak5BsX0xwcg.png)
<span class="figcaption_hack">Building dataset using Dataturks</span>

I found a great plugin that enables batch image downloading on Google Chrome —
this + Dataturks will make building training data a cakewalk . Linked
[here](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en).

You can try doing this with the image_classification tool of dataturks
[here](https://dataturks.com/projects/import?type=IMAGE_CLASSIFICATION). Here
the best feature which this tool provides is , if we have a unstructured dataset
with all the images in a single folder. By manually labelling it with the
classes , you can download a json file which has all the details of the image
with the class embedded in it.Then use the scripts given there for
[keras](https://gist.github.com/sameerg07/e296fe1f8e24933aa6eedd558047278c) and
[tensorflow](https://gist.github.com/sameerg07/4e0337ed2f77845edbb319633bd324e5):

    -------> for tensorflow
    python3 tensorflow_json_parser.py — json_file “flower.json” — dataset_path “Dataset5/”

    -------> for keras
    python3 keras_json_parser.py --json_file "flower.json" --dataset_path "Dataset5/" --train_percentage 80 --validation_percentage 20

#### Training

Now it’s time to train the model. In tensorflow-for-poets-2 folder , there is
folder called scripts which has everything required for re-training of a model.
The **retrain.py** has a special way of cropping and scaling the images which is
too cool .

![](https://cdn-images-1.medium.com/max/1600/1*4PMf03NgVJFaRdhR5k8R1g.png)

Then use the following command to train where the options name itself describe
the required paths to train.:

    python3 -m scripts.retrain \ 
     --bottleneck_dir=tf_files/bottlenecks \ 
     --model_dir=tf_files/models/inception \
     --output_graph=tf_files/retrained_graph.pb \
     --output_labels=tf_files/retrained_labels.txt \
     --image_dir=tf_files/flower_photos

This downloads the inception model and trains the last layer accordingly using
the training folder and the arguments given. I trained it for 4000 steps on a
GCP instance with 12GB Nvidia Tesla k80 and 7GB Vram.

![](https://cdn-images-1.medium.com/max/1600/1*gThLRCa6hu8pFxrKD78LgQ.png)

The training has been done with 80–20 , test- train split and we can see above ,
it gave a test_accuracy of 91% . Now its time to test! We have a .pb file in
**tf_files/ **which can be used to test . The following changes have been added
to the **label_image.py**

    from PIL import Image,ImageDraw,ImageFont
    results = results.tolist()
    image = Image.open(file_name)
    fonttype = ImageFont.truetype(“/usr/share/fonts/truetype/dejav/DejaVuSans.ttf”,18)
     
    draw = ImageDraw.Draw(image)
    draw.text(xy=(5,5),text=str(labels[results.index(max(results))])+”:”+str(max(results)),fill = (255,255,255,128),font = fonttype)
    image.show()
    image.save(file_name.split(“.”)[0]+”1"+”.jpg”)

The above code will help us draw the accuracy on the image being tested and
saves it .confidence percentages fora rodom image shown below

![](https://cdn-images-1.medium.com/max/1600/1*rn-R75wcPnexbe9jKuw7Ag.png)

Few outputs of testing are shown:

![](https://cdn-images-1.medium.com/max/1600/1*Po083pI3sIGxTG7s5ymjRA.jpeg)
<span class="figcaption_hack">a collage of few outputs comprising of all classes</span>

As we have seen the results were really promising enough for the task stated.

### **KERAS:**

Keras is a high level API built on TensorFlow (and can be used on top of Theano
too). It is more user-friendly and easy to use as compared to **Tensorflow. **If
we were a newbie to all this deep learning and wanted to write a new model from
scratch, then Keras is what I would suggest for its ease in both readability and
writability.It can be installed with:

    pip install keras

and even this thing is wrap over tensorflow , so again the CPU v/s GPU
compatibility variations will apply here too.

![](https://cdn-images-1.medium.com/max/1600/0*s3cLmPw09rFq0fDn.png)

Since , we have to carry out the same task of classifying flowers using transfer
learning with inception model , I’ve seen that Keras loads the model in a
standard format like how the APIs are written.

    from keras.applications.inception_v3 import preprocess_input

Keras has a standard format of loading the dataset i.e., instead of giving the
folders directly within a dataset folder , we divide the train and test data
manually and arrange them in the following manner. I have used the same dataset
which I downloaded in the tensorflow section and made few changes as directed
below.

     — Dataset folder -
      — train/ 
           class1/
              — image1
              — image2
           class2/
              — image1
              — image2
      — test/ 
          class1/
              — image1
              — image2
          class2/
              — image1
              — image2

It should look like something below:

![](https://cdn-images-1.medium.com/max/1600/1*YK4JcggiXOH28I9qsc8Nzw.png)

and followed by that train and test should have folders as shown below:

![](https://cdn-images-1.medium.com/max/1600/1*GVMUM4d6NhrO5AZJkkqG2w.png)
<span class="figcaption_hack">TRAIN FOLDER</span>

As, we are now done with the set up of the dataset , it’s time for training ! I
have written down a small piece of code to do the training which goes below:

and this code is neatly written and can be easily understood with the arguments
being passed to the below command:

    python3 inception_train.py 
     — train_dir flower_photos/train \
     — val_dir flower_photos/validation \ 
     — nb_epoch 50 \ 
     — batch_size 10 \ 
     — output_model_file inception_yo1.model

and the training on my GPU took around 1 minute per epoch with 292 steps per
epoch and was trained for 50 epochs (which is very much more ! ) with a batch
size of 10 and a 80–20 data split.

![](https://cdn-images-1.medium.com/max/1600/1*WQyya9IQh5COo3WTyETD_w.png)

Whoop! we are done with training and achieved test_accuracy of ~91% and a loss
of 0.38. The model has been saved as a inception.model file which can be loaded
again and tested . To do that , another script has been written along with
plotting the predicted class on the image and saving it. The testing script goes
as below:

<span class="figcaption_hack">inception_test.py</span>

This script can be tested as :

    python3 -m scripts.label_image — graph=tf_files/retrained_graph.pb — image=rose.jpeg

The predicted confidence percentages over all classes is outputted like:

![](https://cdn-images-1.medium.com/max/1600/1*3TTokRHY-yM5TcR3r8tD6A.png)
<span class="figcaption_hack">[daisy,dandelion,roses,sunflower,tulip]</span>

and below are the few outputs with graph :

![](https://cdn-images-1.medium.com/max/2000/1*-Q8XV88ConT3PenriGt7Qw.jpeg)
<span class="figcaption_hack">tested images with their probability graphs</span>

Finally! you have learnt how to build a powerful classifier using both Keras and
tensorflow. But , which one is best is still a question in our heads! So , let
us do a comparative study only based on this classification task as of now.


#### Prototyping:

If you really want to write a code quickly and build a model , then Keras is a
go. We can build complex models within minutes! The `Model` and the `Sequential`
APIs are so powerful that they wont even give you a sense that you are the
building powerful models due to the ease in using them .

That’s it , a model is ready! Even transfer learning is easy to code in Keras
than in tensorflow. Tensorflow is too tough to code from scratch until you are a
sticky coder .

#### Scratch Coding and flexibility:

As tensorflow is a low-level library when compared to Keras , many new functions
can be implemented in a better way in tensorflow than in Keras for example , any
activation fucntion etc… And also the fine-tuning and tweaking of the model is
very flexible in tensorflow than in Keras due to much more parameters being
available.

#### Training time and processing power:

The above models were trained on the same dataset , we see that Keras takes
loner time to train than tensorflow . Tensorflow finished the training of 4000
steps in 15 minutes where as Keras took around 2 hours for 50 epochs . May be we
cannot compare steps with epochs , but of you see in this case , both gave a
test accuracy of 91% which is comparable and we can depict that keras trains a
bit slower than tensorflow. Apart from this , it makes sense because of
tensorflow being a low level library .

#### Extra features provided:

Tensorflow has a inbuilt debugger which can debug during the training as well as
generating the graphs.

![](https://cdn-images-1.medium.com/max/1600/1*a1Cb6y1jYeOolmC-Qxc6Mw.png)
<span class="figcaption_hack">TensorFlow Debugger snapshot (Source : TensorFlow documentation )</span>

Tensorflow even supports threads and queues to train the heavy tensors
asynchronously! This provides TPUs a better and much faster processing
speeds.Sample code for threads is shown below:

#### Monitoring and Control:

According to my experience in dee learning , I feel tensorflow is highly
suitable for many cases though it is little tough. For example , we can monitor
each and everything very easily such as controlling the weights , gradients of
your network. We can choose which step should be trained and which should not .
This is not that feasible in Keras.The below given line does that magic!

    step = tf.Variable(1, trainable=False, dtype=tf.int32)

### Conclusion:

Anyways , Keras is going to be integrated in tensorflow shortly! So, why go
pythonic?(Keras is pythonic ) . Spend some time and get used to tensorflow is
what I suggest . The classification problem above , if you have followed the
blog and done the steps accordingly , then you will feel that Keras is little
painful and patience killer than tensorflow in many aspects. So , try using
other classes and try training classifers for applications like fake note
detection etc…

Hope this blog would have given you a better insight to what to use when !

**I would love to hear any suggestions or queries. Please write to us at
contact@dataturks.com**

From a quick cheer to a standing ovation, clap to show how much you enjoyed this
story.

### [DataTurks: Data Annotations Made Super Easy](https://hackernoon.com/@dataturks)

Data Annotation Platform. Image Bounding, Document Annotation, NLP and Text
Annotations. #HumanInTheLoop #AI, #TrainingData for #MachineLearning.
