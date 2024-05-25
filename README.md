# Cryptocurrency Wallet Address Classifier

## Project Overview

This project involves the development of a machine learning model to classify cryptocurrency wallet addresses. The classifier is designed to predict the type of cryptocurrency (Bitcoin, Bitcoin Cash, Ethereum, or Litecoin) based on the input wallet address. The project utilizes a pre-trained deep learning model and a label encoder to achieve this task.

## Features

- **Address Classification**: Predicts whether a given cryptocurrency wallet address belongs to Bitcoin, Bitcoin Cash, Ethereum, or Litecoin.
- **User-Friendly Interface**: A Streamlit-based web application that allows users to input a wallet address and receive classification results instantly.
- **Error Handling**: Robust error handling mechanisms to ensure the app operates smoothly even when unexpected issues arise.

## Project Components

1. **Synthetic Dataset Generation**:
   - Generated a synthetic dataset with data augmentation techniques for cryptocurrency address classification.

2. **Feature Extraction**:
   - Extracted relevant features from the wallet addresses, including length, prefix, and character distribution.

3. **Model Training and Comparison**:
   - Trained different machine learning models on the dataset and compared their performance.
   - Selected the best-performing model for further refinement.

4. **Deep Neural Network (DNN) Training**:
   - Trained a Deep Neural Network (DNN) for the final classification task.
   - Saved the trained model (`final.h5`) for deployment.

5. **Streamlit Application**:
   - Built a web application using Streamlit to provide an interactive user interface for address classification.
   - Sidebar input for entering the wallet address and a button to trigger the classification process.
   - Display of classification probabilities for each cryptocurrency type.

6. **Error Handling**:
   - Handled errors related to loading the model and label encoder.
   - Validated input addresses to ensure they are correctly formatted and contain valid prefixes.
   - Provided user-friendly error messages to guide users in case of input errors or other issues.

## Error Handling and Messages

- If the model or label encoder fails to load, the app will display an appropriate error message and stop execution.
- If the input address is invalid or has an unrecognized prefix, the app will inform the user to check the address.
- Any other errors during prediction will be caught and displayed to the user, ensuring a smooth user experience.

## Conclusion

This project demonstrates the application of deep learning for the classification of cryptocurrency wallet addresses. By leveraging a pre-trained model and a user-friendly Streamlit interface, users can easily predict the type of cryptocurrency associated with a given wallet address. The project also emphasizes robust error handling to enhance reliability and user experience. The use of a synthetic dataset with data augmentation, feature extraction, and comparison of different ML models highlights the comprehensive approach taken to develop an effective classification solution.

## Live Demo

You can access the deployed Streamlit app here: [Cryptocurrency Wallet Address Classifier](https://crypto-address-classification-4istoklfwtywwiuyzeqyu2.streamlit.app)
