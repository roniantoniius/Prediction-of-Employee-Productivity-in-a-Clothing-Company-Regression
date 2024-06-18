import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@st.cache_resource
def load_model():
    with open('RandomForest_Productivity_4.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_scalers():
    with open('smv_scaler.pkl', 'rb') as file:
        smv_scaler = pickle.load(file)
    with open('over_time_scaler.pkl', 'rb') as file:
        over_time_scaler = pickle.load(file)
    with open('no_of_workers_scaler.pkl', 'rb') as file:
        no_of_workers_scaler = pickle.load(file)
    return smv_scaler, over_time_scaler, no_of_workers_scaler

st.set_page_config(layout='centered',
                   page_title='Prediksi Produktivitas Buruh Pada Perusahaan Pakaian',
                   page_icon='👕',
                   initial_sidebar_state='expanded')

model = load_model()
smv_scaler, over_time_scaler, no_of_workers_scaler = load_scalers()

st.title('Prediksi Produktivitas Buruh Pada Perusahaan Pakaian')

st.image('gambar.png', caption='Prediksi Produktivitas Buruh')

st.sidebar.title('Informasi Fitur')
st.sidebar.markdown('''
- **Date**: Tanggal
- **Day**: Hari dalam seminggu
- **Quarter**: Kuartal dalam setahun
- **Department**: Nama departemen
- **team_no**: Nomor tim
- **no_of_workers**: Jumlah pekerja dalam tim
- **no_of_style_change**: Jumlah perubahan desain produk
- **targeted_productivity**: Target produktivitas tim per hari
- **Smv**: Waktu standar untuk menyelesaikan tugas
- **Wip**: Jumlah produk yang belum selesai diproduksi
- **over_time**: Waktu tambahan yang digunakan oleh tim
- **Incentive**: Insentif yang diberikan kepada pekerja
- **idle_time**: Waktu idle karena gangguan produksi
- **idle_men**: Jumlah pekerja yang idle karena gangguan produksi
- **actual_productivity**: Tingkat produktivitas pekerja (0-1)
''')

def predict_productivity(inputs):
    input_df = pd.DataFrame([inputs])
    
    # scaling
    input_df['smv'] = smv_scaler.transform(input_df[['smv']])
    input_df['over_time'] = over_time_scaler.transform(input_df[['over_time']])
    input_df['incentive'] = np.log1p(input_df['incentive'])
    input_df['idle_time'] = np.log1p(input_df['idle_time'])
    input_df['no_of_workers'] = no_of_workers_scaler.transform(input_df[['no_of_workers']])
    
    # prediksi
    prediction = model.predict(input_df)
    return prediction[0]

st.header('Masukkan Data')

# form untuk input data
input_data = {
    'team': st.slider('Team', min_value=1, max_value=12, step=1),
    'targeted_productivity': st.slider('Targeted Productivity', min_value=0.0, max_value=1.0, step=0.01),
    'smv': st.slider('SMV', min_value=0.0, max_value=100.0, step=0.1),
    'over_time': st.slider('Over Time', min_value=0, max_value=20000, step=100),
    'incentive': st.slider('Incentive', min_value=0, max_value=5000, step=10),
    'idle_time': st.slider('Idle Time', min_value=0.0, max_value=300.0, step=0.1),
    'no_of_style_change': st.slider('Number of Style Change', min_value=0, max_value=2, step=1, format='%d'),
    'no_of_workers': st.slider('Number of Workers', min_value=0.0, max_value=60.0, step=1.0),
    'day_Saturday': st.slider('Day Saturday', min_value=0, max_value=1, step=1, format='%d'),
    'day_Tuesday': st.slider('Day Tuesday', min_value=0, max_value=1, step=1, format='%d'),
    'department_finishing': st.slider('Department Finishing', min_value=0, max_value=1, step=1, format='%d'),
    'department_finishing2': st.slider('Department Finishing 2', min_value=0, max_value=1, step=1, format='%d'),
    'department_sweing': st.slider('Department Sweing', min_value=0, max_value=1, step=1, format='%d'),
}

if st.button('Predict'):
    prediction = predict_productivity(input_data)
    st.success(f'Predicted Productivity: {prediction:.4f}')
