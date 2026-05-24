import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Amazon Book Analytics Dashboard", layout="wide", page_icon="📚")

# --- 1. LOAD DATA & ML MODEL ---
@st.cache_data # Caches data so your app stays fast when clicking buttons
def load_data():
    df = pd.read_csv("Amazon_BestSelling_Books_500.csv")
    return df

df = load_data()

@st.cache_resource # Caches your model into memory
def load_model():
    with open('book_success_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

try:
    ml_model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# --- APP HEADER ---
st.title(" Amazon Bestseller Book Explorer & Predictor")
st.write("Explore Amazon's top-performing books or leverage machine learning to predict how long a book will remain a bestseller.")

# --- CREATE TWO TABS FOR EXTRA CLEAN UI ---
tab1, tab2 = st.tabs(["Book Finder & Recommendation", " Predict Book Longevity"])

# ==============================================================================
# TAB 1: USER FILTERING & RECOMMENDATIONS (NO ML NEEDED)
# ==============================================================================
with tab1:
    st.header("Search and Filter Bestsellers")
    
    # Structural layout using columns
    col1, col2 = st.columns(2)
    
    with col1:
        selected_category = st.selectbox("Select Category:", df['Category'].unique())
        
        # Pulling publishers sorted alphabetically
        publishers = sorted(df['Publisher'].dropna().unique())
        selected_publisher = st.selectbox("Select Publisher:", ["All"] + list(publishers))

    with col2:
        max_price = float(df['Price (USD)'].max())
        min_price = float(df['Price (USD)'].min())
        selected_price = st.slider("Maximum Budget (USD):", min_price, max_price, max_price)
        
        authors = sorted(df['Author'].dropna().unique())
        selected_author = st.selectbox("Select Author:", ["All"] + list(authors))

    # Apply interactive filters to our DataFrame copy
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    filtered_df = filtered_df[filtered_df['Price (USD)'] <= selected_price]
    
    if selected_publisher != "All":
        filtered_df = filtered_df[filtered_df['Publisher'] == selected_publisher]
    if selected_author != "All":
        filtered_df = filtered_df[filtered_df['Author'] == selected_author]

    # Display results
    st.subheader(f"Found {len(filtered_df)} Matching Books")
    if not filtered_df.empty:
        st.dataframe(
            filtered_df[['Rank', 'Title', 'Author', 'Price (USD)', 'Rating', 'Reviews', 'Weeks on List', 'Publisher']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No books match your current filters. Try relaxing your parameters!")

# ==============================================================================
# TAB 2: MACHINE LEARNING SUCCESS PREDICTOR (ML-BASED)
# ==============================================================================
with tab2:
    st.header("Predict Weeks on Bestseller List")
    st.write("Fill in details for a **theoretical new book release** to let our Random Forest model estimate its shelf-life.")
    
    if not model_loaded:
        st.error("Error: 'book_success_model.pkl' not found in this directory. Please check your filename!")
    else:
        col3, col4 = st.columns(2)
        
        with col3:
            input_price = st.number_input("Book Retail Price (USD):", min_value=0.99, max_value=100.0, value=14.99, step=0.5)
            input_rating = st.slider("Target Customer Rating:", 1.0, 5.0, 4.5, step=0.1)
            input_reviews = st.number_input("Estimated Accumulated Reviews:", min_value=0, max_value=500000, value=10000, step=100)
            input_bsr = st.number_input("Current/Target Amazon BSR (Rank):", min_value=1, max_value=10000, value=150)

        with col4:
            input_category = st.radio("Category Type:", ["Fiction", "Non-Fiction"])
            input_format = st.selectbox("Book Binding Format:", ["Board Book", "Hardcover", "Kindle Edition", "Paperback"])
            input_author_count = st.number_input("How many other bestsellers has this Author written?:", min_value=0, max_value=50, value=2)
            input_publisher_count = st.number_input("How many other bestsellers has this Publisher produced?:", min_value=0, max_value=100, value=10)
            input_years = st.number_input("Years Since Initial Book Launch:", min_value=0, max_value=50, value=1)

        # --- PROCESS INPUT FOR THE ML MODEL ---
        # Map Category back to encoded numeric form (Fiction=0, Non-Fiction=1 depending on your LabelEncoder)
        cat_encoded = 1 if input_category == "Non-Fiction" else 0
        
        # Recreate One-Hot encoded columns for Format (matching our X feature training shapes)
        f_board = 1 if input_format == "Board Book" else 0
        f_hardc = 1 if input_format == "Hardcover" else 0
        f_kindle = 1 if input_format == "Kindle Edition" else 0
        f_paper = 1 if input_format == "Paperback" else 0

        # Construct the final feature array in the exact column order your ML Model expects:
        # ['Price (USD)', 'Rating', 'Reviews', 'Amazon BSR', 'Category_encoded', 
        #  'Format_Board Book', 'Format_Hardcover', 'Format_Kindle Edition', 'Format_Paperback', 
        #  'Author_Book_Count', 'Publisher_Book_Count', 'Year_Since_Release']
        features = np.array([[
            input_price, input_rating, input_reviews, input_bsr, cat_encoded,
            f_board, f_hardc, f_kindle, f_paper, 
            input_author_count, input_publisher_count, input_years
        ]])

        # --- PREDICT BUTTON ---
        if st.button(" Calculate Estimated Performance"):
            prediction = ml_model.predict(features)[0]
            
            st.success("### Prediction Generated!")
            st.metric(
                label="Estimated Weeks on Amazon Bestseller List", 
                value=f"{round(prediction, 1)} Weeks"
            )
            st.info(" Pro-Tip: Increasing the estimated review volume or author clout score drastically alters the longevity scale, mirroring real marketplace trends!")
