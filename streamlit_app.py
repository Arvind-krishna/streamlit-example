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
st.write('''Will be using the **MNIST** dataset to train our neural network.''')
st.write('''The MNIST dataset contains various samples of handwritten digits from 0-9. They are a total of 70,000 images in the dataset, and each has 784 dimensions''')
st.write('''Think of each image as 28x28 grid (=784 total cells). Each cell represents a pixel, and it stores the intensity value for that particular pixel.''')
st.write('''We will now load the MNIST dataset, andf also look at a few samples from the dataset.''')
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

code='''from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', as_frame=False, cache=False) #Loads mnist dataset from sklearn'''

st.code(code, language='python')
st.write('''Enter a number below, and click on "Show Image" to see a sample image of the selected number from the **MNIST** Dataset.''')

number = st.number_input("Enter a number (0-9)", min_value=0, max_value=9, value=0, step=1)

if st.button('Show Image'):
    # Filter for the selected number
    filtered_indices = [i for i, label in enumerate(train_labels) if label == number]

    if filtered_indices:
        # Display the first image found with the selected number
        selected_index = filtered_indices[0]
        st.image(train_images[selected_index],width=200, caption=f"Image of {number}")
    else:
        st.write(f"No images found for the number {number} in the dataset.")







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
st.write('''But before that, we need to determine the test size. This represents the amount of data to be used in the test split.''')
st.write('''For example, lets say we have 10000 images. If we provide a value of 0.25 as test size, it means that the test size is 25% of the total, so it will be 2500 images''')
st.write('''It's also important to experiment with various test sizes, and notice the difference it makes, as test size is plays a key role while training a model.''')
st.write('''Let's now decide the  test size for our neural network.''')
testsplit = st.number_input("Enter a number (0-1)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsplit, random_state=42) #Creating train test split


code1='''from sklearn.model_selection import train_test_split #This package helps split data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size='''+str(testsplit)+''', random_state=42) #Creating train test split'''

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
#####################

# Function to create a simple model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# Input for the number of epochs
num_epochs = st.number_input("Enter the number of epochs", min_value=1, value=5, step=1)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0



##################################
st.code(f"""net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs={num_epochs},
    lr=0.1,
    device=device,
)""",language='python')

st.write(f"Great! now our detective 'net' is ready for training. Since  max_epochs to {num_epochs}, 'net' will train for a total of {num_epochs} cycles/epochs") 


st.code('''net.fit(X_train, y_train);''',language='python')
# Train the model based on user input
if st.button('Train Model'):
    model = create_model()
    model.fit(train_images, train_labels, epochs=num_epochs)
    st.write(f"Hurray! our neural networks has been trained for {num_epochs} epochs.")


st.write('''Awesome, now that our detective has been trained. Now, let's test our detective to see how well it performs!''')

st.subheader('''Step 4 : Testing our detective ''')
st.write('''Let's import a package from sklearn that helps us calculate accuracy score''')
st.code('''from sklearn.metrics import accuracy_score''',language='python')
st.write('''now, let's have 'net' give us predictions for data from the test split''') 
st.code('''y_pred = net.predict(X_test)''',language='python')
st.write('''Ok, so now, 'net' has been tested. let's now evaluate and see its accuracy.''')

st.subheader('''Step 5 : Evaluating our detective ''')
st.write('''Let's now calculate the accuracy of net's predictions vs the actual results''')
st.code('''accuracy_score(y_test, y_pred)''',language='python')
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc=''
if st.button('Evaluate Model'):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    st.code(f"{test_acc}")
    st.markdown(f"Wow, our neural network was able to predict with an accuracy of  {test_acc}! For a network with only one hidden layer, it is not too bad!")


st.markdown('''**Congratulations! You have completed this Chapter on "Handwriting detection using a simple neural network!**''')
st.markdown('''Need help? Please check out the FAQ section of this chapter. Your question may have been answered already.''')
st.markdown('''Can't find the help you're looking for? kindly use a communication channel to reach out to us, and we'll have our experts guide you as soon as possible''')



