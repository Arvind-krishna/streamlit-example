from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


st.title("Vizuara AI Labs")
st.title("Handwritten Text Classification : Numbers")
st.image("image.jpg")
st.write('''Handwriting detection is like a special superpower that helps computers read and understand what we write with our hands!''')

st.write('''It's like teaching a computer to recognize your writing, just like your teacher can!. We will be using a 'neural network' to make this happen. ''')

st.write('''Think of a neural network as a super-smart detective who learns by looking at lots of different handwriting examples. ''')
st.write('''This detective then figures out which letters and words are written in a special code. Once the detective knows the code, it can read what you wrote and even tell you what it says!''')

st.subheader('''Step 1 : Gathering evidence''')

st.write('''Before our detective (the neural network) can become really good at reading handwriting, it needs some high quality evidence/clues, and also some training, just like when you learn to ride a bicycle. ''')
st.write('''In this chapter, we will be training our detective to analyse and detect handwritten numbers. Lets gather some data containing handwritten numbers, which could help us train our detective''')
st.write('''Will be using the **MNIST** dataset to train our neural network''')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

number = st.number_input("Enter a number (0-9)", min_value=0, max_value=9, value=0, step=1)

if st.button('Show Image'):
    # Filter for the selected number
    filtered_indices = [i for i, label in enumerate(train_labels) if label == number]

    if filtered_indices:
        # Display the first image found with the selected number
        selected_index = filtered_indices[0]
        st.image(train_images[selected_index], caption=f"Image of {number}")
    else:
        st.write(f"No images found for the number {number} in the dataset.")




code='''from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', as_frame=False, cache=False) #Loads mnist dataset from sklearn'''

st.code(code, language='python')


st.write('''Awesome!. We've now loaded the dataset of handwritten numbers from 0-9. That's a great start!. now, to train our detective.''')

st.markdown("**Ok, Now how do we ensure our detective is ready to efficiently detect handwriting?**")
choices = ["Train the detective!", "Test the detective!", "Train, and then test the detective!"]
correct_answer = "Train, and then test the detective!"

# Create a multiple-choice question using radio buttons
user_answer = st.radio("Select your answer:", choices)

# Check if the user's answer is correct and display feedback
if user_answer == correct_answer:
    st.markdown("**Excellent! We need to train the detective, and also test, to see how good our detective really is!** :green_heart:")
elif user_answer == "Train the detective!":
    st.markdown("**Not quite. Something is missing!** :orange_heart:") 
else :  
    st.markdown("**Oops, our detective hasn't been trained yet** :orange_heart:") 



st.subheader('''Step 2 : Creating a Training and Testing framework ''')

st.write('''In order to train our detective, We provide it lots and lots of different examples of handwriting, like the digits '1,' '3,' and '6', written in various ways ''')
st.write('''It practices and practices until it gets better at recognizing them.''')
st.write('''But remember, a good detective needs to be tested to make sure they're really good at their job.''')
st.write('''So, after all the training, we give our detective a special test. ''')
st.write('''We show it some new handwriting that it has never seen before, numbers written in ways he detective hasnt directly seen in the training ''')
st.write('''Our detective tries its best to read them. If it does a great job, that means our neural network is ready to help us accurately read handwriting. ''')
st.write('''If not, we give it some more practice until it gets better at it. ''')
st.write('''This testing part is like the detective showing how well it learned from the training, just like when you show how good you've become at riding your bike.''')
st.write('''So, with practice and testing, our neural network becomes a real handwriting expert!"''')
st.write('''We will first be splitting our data, into a training part, and a testing part''')


code1='''from sklearn.model_selection import train_test_split #This package helps split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) #Creating train test split'''

st.code(code1, language='python')
st.markdown("""
**Did you know?**

Why do we see random_state="42" a lot? 
It is actually a reference from the movie "The Hitchhiker's Guide to the Galaxy". It soon became a programming convention that caught on.

In the movie, when a supercomputer is asked the answer to “the Ultimate Question of Life, the Universe, and Everything." It thinks for a 
really long time, and responds with "42"
""")

st.subheader('''Step 3 : Grooming our detective ''')

st.write('''"Okay, so now, we're getting ready to teach a computer how to recognize pictures. Think of it like training a robot to tell us what's in a picture. To do this, we need to make the pictures a special shape so the computer can understand them. It's like putting the pictures in a magic box.

Each picture is like a tiny puzzle made up of squares (pixels). Our magic box needs to know how many squares are in each picture, and we also have to tell it how many pictures we're going to show. For the pictures we're using today, each picture is 28 squares high and 28 squares wide.''')
st.image("grid.png")

st.write('''We're now going to create a simple, fully connected neural network with one hidden layer. ''')
st.write('''The input layer has 784 dimensions (=28x28, based on the input picture), hidden layer has 98 (= 784 / 8) and the output layer 10 neurons, representing digits 0 - 9.''')
st.write('''Let's quickly try to understand how the layers work, using a simple real-world example:''')
note1='''The first layer helps the robot understand the picture. It's like the robot's eyes. It looks at the picture and tries to recognize what's in it.

The second layer helps the robot think and make decisions. It's like the robot's brain. It thinks really hard about the picture to figure out what's actually inside.

The third layer helps the robot make its final decision. It's like the robot's mouth. It says, 'I think it's 4!' or 'I think it's 7!'''
st.write(note1)
st.write('''Let's now import the PyTorch packages we'll be using, set up the device, and also the dimensions for our Neural Network.''')
code2='''import torch 
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu' #switching device to 'cuda'(gpu) if available, else 'cpu'

mnist_dim = X.shape[1]
hidden_dim = int(mnist_dim/8)
output_dim = len(np.unique(mnist.target))

mnist_dim, hidden_dim, output_dim
(784, 98, 10)'''
st.code(code2, language='python')
st.write('''Now, let's create a neural network in PyTorch''')
code3='''class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=0.5,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X'''
st.code(code3, language='python')
st.write('''Does the above code confuse you? Don't know what's going on? 

Dont worry! We'll explore the steps/usage in a simplified manner.

**ReLU (Rectified Linear Unit)**: Think of 'ReLU' as the robot's way of getting excited when it sees something interesting in a picture. It's like when you get really happy when you see your favorite toy. Our robot's 'ReLU' move makes it light up with excitement and focus on the exciting parts of the picture.

**Dropout**: Sometimes, our robot wants to be really careful. It's like when you're crossing a tricky bridge, and you take tiny steps to make sure you don't fall. Our robot does something similar with a move called 'dropout.' When it's not very sure about the picture, it uses 'dropout' to slow down and think more. This helps it avoid making mistakes. So, 'dropout' is like our robot's safety move.

**Softmax**: When our robot is confident and ready to tell us what's in the picture, it uses 'softmax.' It's like when you're sure about the answer to a question and raise your hand in class. Our robot speaks up and confidently says, 'I think it's 8!' or 'I think it's 7!' It doesn't guess – it uses 'softmax' to make a smart and confident decision.

**Forward**: To make all of this happen, our robot has a special step called 'forward.' It's like a magic spell that combines 'dropout' and 'softmax' with its thinking and seeing abilities. When the robot looks at the picture, it thinks about it, uses 'dropout' to be careful, and then speaks up with 'softmax' to tell us what it sees. 'Forward' is like the robot's secret recipe for playing the picture game.

With these three moves, our robot becomes a handwriting detection champion. It's cautious when it needs to be and confident when it knows the answer, all thanks to 'dropout,' 'softmax,' and 'forward'!"''')

st.write('''Before moving on to the next step, let's have a quick trivia to recap''')

st.write("""**In a secret code game, you have a message written in Invisible Ink. You need a special flashlight to see the message. 
Is the hidden layer in a neural network like the invisible ink, the flashlight, or the message that appears when you use the flashlight?**""")
choices = ["Invisible Ink", "Flashlight", "Message"]
correct_choice = "Message"


user_answer = st.radio("", choices)


correct_message = """**Awesome, Great job!**
The hidden layer in a neural network is like the message that appears when you use the flashlight.
Just as the message becomes visible when you shine the special flashlight on the invisible ink, the hidden layer in a neural network contains important information that becomes useful and meaningful when the network processes it.
So, it's a bit like the magic behind the scenes that helps the network understand and make decisions :green_heart:.
"""

incorrect_message = "**Oops, not really** :orange_heart:"


if user_answer == correct_choice:
    st.markdown(correct_message)
else:
    st.markdown(incorrect_message)


st.subheader('''Step 3 : Building and training our detective ''')
st.write('''next, we'll import NeuralNetClassifier from skorch, and create our detective. skorch allows to use PyTorch's networks in the SciKit-Learn setting''')
st.code('''from skorch import NeuralNetClassifier''', language='python')
st.write('''Now, let's build our detective,whom we shall be called "net"''')
st.code('''net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
)''',language='python')

st.write('''Great! now our detective 'net' is ready for training. Since we've set max_epochs to 20, 'net' will train for a total of 20 cycles/epochs''') 
st.code('''net.fit(X_train, y_train);''',language='python')
op=''' epoch    train_loss    valid_acc    valid_loss     dur
-------  ------------  -----------  ------------  ------
      1        0.8387       0.8800        0.4174  3.9125
      2        0.4332       0.9103        0.3133  0.9897
      3        0.3612       0.9233        0.2684  1.3494
      4        0.3233       0.9309        0.2317  1.3793
      5        0.2938       0.9353        0.2173  0.9992
      6        0.2738       0.9390        0.2039  0.9964
      7        0.2600       0.9454        0.1868  1.0119
      8        0.2427       0.9484        0.1757  1.6899
      9        0.2362       0.9503        0.1683  1.0590
     10        0.2226       0.9512        0.1621  1.0045
     11        0.2184       0.9529        0.1565  0.9868
     12        0.2090       0.9541        0.1508  0.9842
     13        0.2067       0.9570        0.1446  0.9813
     14        0.1978       0.9570        0.1412  1.3288
     15        0.1923       0.9582        0.1392  1.3540
     16        0.1889       0.9582        0.1342  1.1838
     17        0.1855       0.9612        0.1297  1.1465
     18        0.1786       0.9613        0.1266  0.9851
     19        0.1728       0.9615        0.1250  0.9879
     20        0.1698       0.9613        0.1248  1.0151'''
st.code(op,language='python')
st.write('''Awesome, our detective has been trained. Now, let's test our detective to see how well it performs!''')

st.subheader('''Step 4 : Testing our detective ''')
st.write('''Let's import a package from sklearn that helps us calculate accuracy score''')
st.code('''from sklearn.metrics import accuracy_score''',language='python')
st.write('''now, let's have 'net' give us predictions for data from the test split''') 
st.code('''y_pred = net.predict(X_test)''',language='python')
st.write('''Ok, so now, 'net' has been tested. let's now evaluate and see its accuracy.''')

st.subheader('''Step 5 : Evaluating our detective ''')
st.write('''Let's now calculate the accuracy of net's predictions vs the actual results''')
st.code('''accuracy_score(y_test, y_pred)''',language='python')
st.code('''0.9631428571428572''',language='python')
st.markdown('''Wow, net was able to predict with an accuracy of  about **96%**! For a network with only one hidden layer, it is not too bad!''')


st.markdown('''**Congratulations! You have completed this Chapter on "Handwriting detection using a simple neural network!**''')
st.markdown('''Need help? Please check out the FAQ section of this chapter. Your question may have been answered already.''')
st.markdown('''Can't find the help you're looking for? kindly use a communication channel to reach out to us, and we'll have our experts guide you as soon as possible''')



