import streamlit as st
import pandas as pd
import pickle
import sklearn 
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image

dataset=pd.read_csv(r'C:\Users\CODED\OneDrive\Documents\housepredictionproject\nigeria_houses_data.csv')
st.set_page_config(page_title='ZEE & SODIQ REALTY PROPERTIES',page_icon='House',layout='wide')
st.header('ZEESOD REALTY & PROPERTIES')
st.sidebar.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\LOGO.png")

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\region.pkl",'rb') as  file:
   region_encoder=pickle.load(file)
with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\title.pkl",'rb') as  file:
    title_encoder=pickle.load(file)
with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\state.pkl",'rb')as file:
    state_encoder=pickle.load(file)
with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\town.pkl",'rb')as file:
    town_encoder=pickle.load(file)
with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\Apartment_type.pkl",'rb')as file:
    Apartment_type_encoder=pickle.load(file)

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\bungalow.pkl", 'rb') as file:
    bungalow_encoder=pickle.load(file)

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\duplex.pkl", 'rb') as file:
    duplex_encoder = pickle.load(file)

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\encoders\flat.pkl", 'rb') as file:
    flat_encoder = pickle.load(file)

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\gbr_model.pkl",'rb')as file:
   gbr_model=pickle.load(file)

with open(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\scalar.pkl",'rb') as file:
   scalar = pickle.load(file)

def app():
    st.markdown('##### Let Help You Get Your Dream Home')
    Apartment_type,State,Town,Title=st.columns(4)
    with Apartment_type:
        st.subheader('Apartment')
        apartment = st.selectbox('Apartment Type',Apartment_type_encoder.classes_)
    with State:
        st.subheader('State')
        state = st.selectbox('State', state_encoder.classes_)
    with Town:
        st.subheader('Location')
        town = st.selectbox('Location', town_encoder.classes_)
    with Title:
        st.subheader('Title')
        title = st.selectbox('Title', title_encoder.classes_)
    st.write('---------------------------------------------------------------------------------------------')

    image1,image2,image3=st.columns(3)
    with image1:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image1.jpeg")
        st.write('comfort apartment from $119-$20,000')
    with image2:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image2.jpeg")
        st.write('Exquisite Apartment from $28,000 -$118,421.05')
    with image3:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image3.jpeg")
        st.write('standard luxury and serviced Apartment from $13,157.89 -$526,315.79')

    #st.write('------------------------------------------------------------------------------------------------')
    image4,image5,image6=st.columns(3)
    with image4:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image4.jpg")
        st.write('A storey building apartment with garden')
    with image5:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image5.jpg")
        st.write('Exquisite storey building Apartment')
    with image6:
        st.image(r"C:\Users\CODED\OneDrive\Documents\housepredictionproject\image6.jpg")
        st.write('standard bungalow luxury and serviced Apartment')



    #Apartment_type,State,Town,Title=st.columns(4)
    #with Apartment_type:
        #st.subheader('Apartment')
        #apartment = st.selectbox('Apartment Type',Apartment_type_encoder.classes_)
    #with State:
        #st.subheader('State')
        #state = st.selectbox('State', state_encoder.classes_)
    #with Town:
        #st.subheader('Location')
        #town = st.selectbox('Location', town_encoder.classes_)
    #with Title:
        #st.subheader('Title')
        #title = st.selectbox('Title', title_encoder.classes_)

    bungalow = st.sidebar.selectbox("Bungalow", bungalow_encoder.classes_)
    duplex = st.sidebar.selectbox("Duplex", duplex_encoder.classes_)
    flat = st.sidebar.selectbox("Flat", flat_encoder.classes_)
    region = st.sidebar.selectbox("Region", region_encoder.classes_)

    bedroom = st.sidebar.slider('Bedrooms', min_value = 1   , max_value = 9)
    bathroom = st.sidebar.slider('Bathrooms', min_value = 1   , max_value = 9)
    toilets = st.sidebar.slider('Toilets', min_value =1, max_value = 5)
    parking_space =  st.sidebar.slider('Parking Space', min_value =1, max_value = 5)

    apartment = Apartment_type_encoder.transform([apartment])[0]
    state = state_encoder.transform([state])[0]
    town = town_encoder.transform([town])[0]
    title = title_encoder.transform([title])[0]
    region = region_encoder.transform([region])[0]
    bungalow = bungalow_encoder.transform([bungalow])[0]
    duplex = duplex_encoder.transform([duplex])[0]
    flat = flat_encoder.transform([flat])[0]

    data = {'bedrooms': bedroom, 'bathrooms': bathroom, 'toilets': toilets, 
            'parking_space': parking_space, 'title': title, 'town': town, 'state': state, 
            'duplex_apartment': duplex,  'bungalow_apartment': bungalow, 'flat_apartment': flat,
             'region': region, 'Apartment_type': apartment,
            }
    
    df = pd.DataFrame(data, index= [0])
    scaled_inputs = scalar.transform(df)


    if st.button('Predict Price of House'):
        st.subheader('Price')
        predicted_price = gbr_model.predict(scaled_inputs)[0]
        st.success(f'The predicted price of the house is: {predicted_price:.2f}')

app()
