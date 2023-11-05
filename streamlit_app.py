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
    st.markdown("**Correct! We need to train the detective, and also test, to see how good our AI detective really is!** :green_heart:")
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

st.write('''Now, we want to split our pictures into two groups: one to help our robot learn (we call it 'XCnn_train'), and another group for testing how well it learned (we call it 'XCnn_test'). It's like having a practice session before a big game! We also have some labels to tell the robot what's in each picture.''')




