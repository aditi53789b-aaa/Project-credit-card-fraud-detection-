import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='CTGAN Fraud Detection', layout='centered')

st.title('🛡️ CTGAN Synthetic Fraud Detection')
st.write('This demo generates synthetic transaction data for fraud detection research.')

# Generate sample synthetic data
data = pd.DataFrame({
    'TransactionAmount': np.random.randint(100, 5000, 10),
    'TransactionType': np.random.choice(['UPI', 'Card', 'NetBanking'], 10),
    'FraudProbability': np.round(np.random.rand(10), 2)
})

st.subheader('Generated Synthetic Transactions')
st.dataframe(data)

st.success('Synthetic data generated successfully!')