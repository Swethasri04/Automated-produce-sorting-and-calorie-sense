import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

# Load the model
model = load_model('FV.h5')

# Labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = [
    'Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple',
    'Pomegranate', 'Watermelon'
]

vegetables = [
    'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn',
    'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas', 'Potato',
    'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato',
    'Turnip'
]

# Function to fetch calories
def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Unable to fetch the Calories")
        return None

# Streamlit UI
st.title("Fruit & Vegetable Classifier 🍎🥦")
st.text("Upload an image of a fruit or vegetable to identify it and get calorie info.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    img = image.resize((150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    pred = np.argmax(result)
    prediction = labels[pred]
    
    st.success(f"Prediction: {prediction.capitalize()}")

    # Check if it's fruit or vegetable
    if prediction.capitalize() in fruits:
        st.info("Category: Fruit")
    elif prediction.capitalize() in vegetables:
        st.info("Category: Vegetable")
    else:
        st.warning("Category: Unknown")

    # Fetch and display calorie info
    calories = fetch_calories(prediction)
    if calories:
        st.write(f"**Calories in 100g of {prediction.capitalize()}:** {calories}")
