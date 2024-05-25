import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model
model = load_model('final.h5')

# Define the feature extraction function
def extract_features(address):
    length = len(address)
    if address.startswith('bc1'):
        prefix = 'bc1'
    elif address.startswith('0x'):
        prefix = '0x'
    elif address.startswith('ltc1'):
        prefix = 'ltc1'
    elif address.startswith('bitcoincash:'):
        prefix = 'bitcoincash:'
    else:
        prefix = address[0]
    
    char_distribution = {char: address.count(char) for char in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    feature_vector = [length, prefix] + [char_distribution.get(char, 0) for char in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
    
    return feature_vector

label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Define the Streamlit app
def main():
    st.title("Cryptocurrency Wallet Address Classifier")
    st.sidebar.header("User Input")

    # User input for address
    address_input = st.sidebar.text_input("Enter the cryptocurrency wallet address:")

    # Button to make prediction
    if st.sidebar.button("Classify"):
        if address_input:
            # Preprocess the input address and extract features
            features = extract_features(address_input)

            # Extract and encode the prefix
            prefix = features[1]
            encoded_prefix = label_encoder.transform([prefix])[0]
            
            # Combine encoded prefix with other features
            combined_features =  [encoded_prefix] + [features[0]] + features[2:]

            # Convert features to numpy array and reshape
            features_np = np.array(combined_features).reshape(1, -1)

            # Make prediction
            probabilities = model.predict(features_np)[0]

            # Decode prediction
            crypto_types = ['Bitcoin', 'Bitcoin Cash', 'Ethereum', 'Litecoin'] 
            st.write("Prediction Probabilities:")
            for crypto, prob in zip(crypto_types, probabilities):
                st.write(f"{crypto}: {prob:.4f}")
        else:
            st.error("Please enter a valid cryptocurrency wallet address.")

# Run the app
if __name__ == "__main__":
    main()
