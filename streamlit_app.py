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

st.write("Ok, Now how do we ensure our detective is ready to efficiently detect handwriting?")
choices = ["Train the detective!", "Test the detective!", "Train, and then test the detective!"]
correct_answer = "Train, and then test the detective!"

# Create a multiple-choice question using radio buttons
user_answer = st.radio("Select your answer:", choices)

# Check if the user's answer is correct and display feedback
if user_answer == correct_answer:
    st.markdown("**Correct! We need to train the detective, and also Test, to see how good he has become!** :green_heart:")
elif user_answer == "Train the detective!":
    st.markdown("**Not quite. Something is missing! ** :orange_heart:") 
else :    



st.subheader('''Step 2 : Training and Testing our AI detective''')

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


