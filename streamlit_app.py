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

st.subheader('''Step 1 : Training our Handwriting detective''')

st.write('''"Before our detective (the neural network) can become really good at reading handwriting, it needs some training, just like when you learn to ride a bicycle. ''')
st.write('''We show it lots and lots of different handwritten examples of the numbers from 0-9 This will be our dataset. It practices and practices until it gets better at recognizing them. ''')
st.write('''This part is like the detective's training.''')

st.write('''Here's a small activity. Pick a color from below, and observe the numbers.''')
selected_color = st.color_picker("Pick a color", "#FF5733")
st.subheader(f'<span style="color: {selected_color};">0123456789</span>', unsafe_allow_html=True)

code='''from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', as_frame=False, cache=False) #Loads mnist dataset from sklearn'''

st.code(code, language='python')

st.write('''Awesome!. We've now loaded the dataset which 

with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))
