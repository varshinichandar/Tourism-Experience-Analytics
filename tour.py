import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Set the title of the Streamlit app
st.title("Tourism Recommender App")

# Load data and models
@st.cache_data
def load_data():
    # Load data from Excel or CSV files
    trans = pd.read_excel("Transaction.xlsx").dropna()
    user = pd.read_excel("User.xlsx").dropna()
    city = pd.read_excel("City.xlsx")
    item = pd.read_excel("Item.xlsx")
    upd_item = pd.read_excel("Updated_Item.xlsx")
    type_df = pd.read_excel("Type.xlsx")

    # Merge dataframes
    total_item = pd.concat([item, upd_item]).drop_duplicates('AttractionId', keep='last')
    city['CityName'] = city['CityName'].str.title().str.strip()
    trans = trans[trans['Rating'].between(1, 5)]

    user_full = user.merge(city, on='CityId', how='left')
    data = trans.merge(user_full, on='UserId', how='left').merge(total_item, on='AttractionId', how='left').merge(type_df, on='AttractionTypeId', how='left')

    return data

data = load_data()
st.success("Data Loaded!")

# Load pre-trained models
clf = joblib.load("visit_mode_model.pkl")
reg = joblib.load("rating_model.pkl")

# Step 1: User Input for Location and Preferred Visit Mode
st.header("Enter Your Details")

city_opt = st.selectbox("Select City", data['CityName'].dropna().unique())
type_opt = st.selectbox("Select Attraction Type", data['AttractionType'].dropna().unique())

# Get user details for visit mode prediction
input_cls = pd.DataFrame({
    'CityName': [city_opt], 
    'AttractionType': [type_opt]
})

if st.button("Predict Visit Mode"):
    input_cls_encoded = pd.get_dummies(input_cls)
    feature_names = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else input_cls_encoded.columns
    input_cls_encoded = input_cls_encoded.reindex(columns=feature_names, fill_value=0)
    pred = clf.predict(input_cls_encoded)
    st.success(f"Predicted Visit Mode: {pred[0]}")

# Step 2: Recommend Attractions Based on User Profile and Transaction History
st.header("Recommend Attractions Based on Your Profile")

user_id = st.number_input("Enter User ID", int(data['UserId'].min()), int(data['UserId'].max()), step=1)

# Filter the data based on selected city
city_filtered_data = data[data['CityName'] == city_opt]

# Create a user-item rating matrix for the selected city
rating_mat = city_filtered_data.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)

# Convert the rating matrix to sparse format for memory efficiency
rating_mat_sparse = csr_matrix(rating_mat)

# Compute cosine similarity between users
sim_users = cosine_similarity(rating_mat_sparse, dense_output=False)

# Map user IDs to indices in the rating matrix
user_id_to_index = {user_id: idx for idx, user_id in enumerate(rating_mat.index)}

# Get similar users and recommend attractions
if user_id not in user_id_to_index:
    st.error(f"User ID {user_id} not found in the selected city.")
else:
    user_index = user_id_to_index[user_id]
    user_similarities = sim_users[user_index].toarray().flatten()
    
    # Get the top 5 most similar users
    sorted_similar_users = np.argsort(user_similarities)[::-1][:5]
    
    recommended_attractions = set()
    for similar_user in sorted_similar_users:
        user_ratings = rating_mat.iloc[similar_user]
        top_attractions = user_ratings[user_ratings > 0].sort_values(ascending=False).head(5).index.tolist()
        recommended_attractions.update(top_attractions)
    
    recommended_attractions = list(recommended_attractions)
    
    if recommended_attractions:
        st.write("Recommended Attractions:")
        st.write(recommended_attractions)
    else:
        st.error("No recommended attractions found.")

# Step 3: Visualizations
st.header("Popular Attractions and User Segments")

# Visualize the top 10 most popular attractions in the selected city
city_attractions = city_filtered_data['Attraction'].value_counts().head(10)
fig, ax = plt.subplots()
city_attractions.plot(kind='bar', ax=ax)
ax.set_title("Top 10 Most Popular Attractions")
ax.set_xlabel("Attraction")
ax.set_ylabel("Number of Visits")
st.pyplot(fig)

# Visualize top regions for tourism
top_regions = city_filtered_data['CityName'].value_counts().head(10)
fig, ax = plt.subplots()
top_regions.plot(kind='bar', ax=ax, color='teal')
ax.set_title("Top 10 Regions for Tourism")
ax.set_xlabel("Region")
ax.set_ylabel("Number of Visitors")
st.pyplot(fig)

# Visualize user segments (e.g., based on Visit Mode or Attraction Type)
user_segments = data['AttractionType'].value_counts()
fig, ax = plt.subplots()
user_segments.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=sns.color_palette("Set2", len(user_segments)))
ax.set_title("User Segments by Attraction Type")
st.pyplot(fig)
