from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st


st.title("Vizuara AI Labs")
st.title("Handwritten Text Classification : Numbers")
st.image("image.jpg")
st.write('''Handwriting detection is like a special superpower that helps computers read and understand what we write with our hands!''')

st.write('''It's like teaching a computer to recognize your writing, just like your teacher can!. We will be using a 'neural network' to make this happen. ''')

st.write('''Think of a neural network as a super-smart detective who learns by looking at lots of different handwriting examples. ''')
st.write('''This detective then figures out which letters and words are written in a special code. Once the detective knows the code, it can read what you wrote and even tell you what it says!''')

st.subheader('''Step 1 : Gathering evidence''')

st.write('''"Before our detective (the neural network) can become really good at reading handwriting, it needs some high quality evidence/clues, and also some training, just like when you learn to ride a bicycle. ''')



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
    st.markdown("**Excellent! We need to train the detective, and also test, to see how good our AI detective really is!** :green_heart:")
elif user_answer == "Train the detective!":
    st.markdown("**Not quite. Something is missing!** :orange_heart:") 
else :  
    st.markdown("**Oops, our detective hasn't been trained yet** :orange_heart:") 



st.subheader('''Step 2 : Creating a Training and Testing framework ''')

st.write('''Before our AI detective can become really good at reading handwriting, it needs some training, just like when you learn to ride a bicycle. ''')
st.write('''We show it lots and lots of different examples of handwriting, like the letters 'A,' 'B,' and 'C.' ''')
st.write('''It practices and practices until it gets better at recognizing them. This part is like the detective's training.''')
st.write('''But remember, a good detective needs to be tested to make sure they're really good at their job.''')
st.write('''So, after all the training, we give our detective a special test. ''')
st.write('''We show it some new handwriting that it has never seen before, like letters 'X,' 'Y,' and 'Z.' ''')
st.write('''Our detective tries its best to read them. If it does a great job, that means our neural network is ready to help us accurately read handwriting. ''')
st.write('''If not, we give it some more practice until it gets better at it. ''')
st.write('''This testing part is like the detective showing how well it learned from the training, just like when you show how good you've become at riding your bike.''')
st.write('''So, with practice and testing, our neural network becomes a real handwriting expert!"''')
st.write('''We will first be splitting our data, into a training part and testing part''')


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

st.write('''"Okay, imagine we're getting ready to teach a computer how to recognize pictures. Think of it like training a robot to tell us what's in a picture. To do this, we need to make the pictures a special shape so the computer can understand them. It's like putting the pictures in a magic box.

Each picture is like a tiny puzzle made up of squares (pixels). Our magic box needs to know how many squares are in each picture, and we also have to tell it how many pictures we're going to show. For the pictures we're using today, they're all in black and white, like drawings. So, we say there's just one color channel (because it's only black and white), and each picture is 28 squares high and 28 squares wide.''')
st.image("grid.png")

st.write('''We're now going to create a simple, fully connected neural network with one hidden layer. ''')
st.write('''The input layer has 784 dimensions (=28x28, based on the input grid), hidden layer has 98 (= 784 / 8) and output layer 10 neurons, representing digits 0 - 9.''')
st.write('''Let's quickly try to understand how the layers work, using a simple real-world example:''')
note1='''The first layer helps the robot understand the picture. It's like the robot's eyes. It looks at the picture and tries to recognize what's in it.

The second layer helps the robot think and make decisions. It's like the robot's brain. It thinks really hard about the picture to figure out what's inside.

The third layer helps the robot make its final decision. It's like the robot's mouth. It says, 'I think it's a cat!' or 'I think it's a dog!'''
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

**Softmax**: When our robot is confident and ready to tell us what's in the picture, it uses 'softmax.' It's like when you're sure about the answer to a question and raise your hand in class. Our robot speaks up and confidently says, 'I think it's a cat!' or 'I think it's a dog!' It doesn't guess – it uses 'softmax' to make a smart and confident decision.

**Forward**: To make all of this happen, our robot has a special step called 'forward.' It's like a magic spell that combines 'dropout' and 'softmax' with its thinking and seeing abilities. When the robot looks at the picture, it thinks about it, uses 'dropout' to be careful, and then speaks up with 'softmax' to tell us what it sees. 'Forward' is like the robot's secret recipe for playing the picture game.

With these three moves, our robot becomes a picture game champion. It's cautious when it needs to be and confident when it knows the answer, all thanks to 'dropout,' 'softmax,' and 'forward'!"''')





