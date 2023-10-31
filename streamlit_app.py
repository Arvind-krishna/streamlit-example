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

question = "Ok, What should we do next?"
choices = ["Train the detective!", "Test the detective!", "Train, and then test the detective!"]
correct_answer = "Train, and then test the detective"

# Create a multiple-choice question using radio buttons
user_answer = st.radio("Select your answer:", choices)

# Check if the user's answer is correct and display feedback
if user_answer == correct_answer:
    st.markdown("**Correct!** :green_heart:")
else:
    st.markdown("**Not quite.** :orange_heart:")


st.subheader('''Step 1 : Gathering evidence''')

st.write('''We show it lots and lots of different handwritten examples of the numbers from 0-9 This will be our dataset. It practices and practices until it gets better at recognizing them. ''')
st.write('''This part is like the detective's training.''')


