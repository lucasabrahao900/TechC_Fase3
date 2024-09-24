import streamlit as st
import pickle

# Load the pickled model
#with open('your_model.pkl', 'rb') as f:
#    model = pickle.load(f)

# Define the Streamlit app
def main():
    st.title("Prediction App")

    # Input fields for the three variables
    var1 = st.number_input("Variable 1")
    var2 = st.number_input("Variable 2")
    var3 = st.number_input("Variable 3")

    # Button to trigger prediction
    if st.button("Predict"):
        # Create a list of input values
        input_data = [[var1, var2, var3]]

        # Make the prediction
        #prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"Prediction: {True}")

if __name__ == '__main__':
    main()