# Tree-Species-Classification

This project focuses on classifying tree species using machine learning techniques. It utilizes a dataset containing various tree features such as height, diameter, leaf area, and possibly environmental attributes like soil type or location. The objective is to train a predictive model that can accurately identify the species of a tree based on these features. Additionally, the project includes a user-friendly Streamlit web application that allows users to upload new data and receive real-time predictions. This tool can be useful for botanists, environmental researchers, or forestry departments in identifying and cataloging tree species efficiently.

Learning Objectives
To develop an AI-powered application for tree species identification and recommendation.
To integrate diverse machine learning models (Nearest Neighbors and Convolutional Neural Networks) into a unified platform.
To create a user-friendly web interface using Streamlit for accessibility.
To implement robust data and model loading mechanisms with error handling.
To provide insights into tree distribution and suitability based on geographical and physical attributes.

GOAL
To empower tree enthusiasts, urban planners, environmentalists, and researchers with a comprehensive, data-driven tool to:
Accurately identify tree species from images.
Receive recommendations for suitable tree species based on specific location and environmental parameters.
Discover common geographical locations where particular tree species thrive.
Foster better understanding and management of tree populations for sustainable environments.

Tools and Technology Used
Python: Core programming language.
Streamlit: For building interactive web applications.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations, especially array handling.
Collections (Counter): For efficient counting of elements.
Joblib: For efficient loading of pre-trained Scikit-learn models (Scaler, Nearest Neighbors).
TensorFlow & Keras: For building, training, and loading the Convolutional Neural Network (CNN) model for image classification.
PIL (Pillow): For image processing and manipulation.

Methodology
1. Data & Model Loading
Efficient Caching: Utilized Streamlit's @st.cache_data and @st.cache_resource decorators for fast, one-time loading of the tree dataset (tree_data.pkl) and pre-trained models (scaler.joblib, nn_model.joblib, basic_cnn_tree_species.h5).
Robust Error Handling: Implemented try-except blocks to gracefully handle FileNotFoundError and other exceptions during data and model loading, providing user-friendly error messages.

2. Tree Recommendation by Location (Nearest Neighbors)
Input Features: Takes user inputs for Latitude, Longitude, Diameter, Native Status, City, and State.
Categorical Encoding: Converts selected categorical features (Native Status, City, State) into numerical codes using the categories learned from the dataset.
Feature Scaling: Applies a pre-trained StandardScaler to normalize numerical input features.
Similarity Search: Employs a pre-trained NearestNeighbors model to find the 5 most similar tree entries in the dataset based on the scaled input features.
Species Aggregation: Counts the occurrences of common names among the nearest neighbors to recommend the top 5 most frequently found species.

3. Finding Locations for a Tree Species
Species Selection: Allows users to select a specific tree species from a dropdown list.
Data Filtering: Filters the main dataset to include only records for the selected species.
Location Aggregation: Groups the filtered data by 'city' and 'state' and counts the occurrences to identify the most common locations.
Display: Presents the top 10 common locations in a clear DataFrame format.

4. Image-based Tree Identification (Convolutional Neural Network - CNN)
Image Upload: Enables users to upload tree images (JPG, JPEG, PNG).
Image Preprocessing: Resizes uploaded images to a standard 224x224 pixels and normalizes pixel values to [0, 1].
Model Prediction: Utilizes a pre-trained CNN model (basic_cnn_tree_species.h5) to predict the tree species from the processed image.
Confidence & Top Predictions: Displays the predicted species with its confidence score and lists the top 3 alternative predictions with their respective confidences.

Integrated Location Search: Automatically fetches and displays common locations for the identified tree species, enhancing the utility of the image recognition feature.

Problem Statement
Difficulty in Tree Identification: Non-experts often struggle to accurately identify tree species, which is crucial for gardening, urban planning, and ecological studies.
Lack of Location-Specific Tree Guidance: It's challenging to determine which tree species are best suited for a particular geographical location or to find where a specific species commonly grows.
Manual Data Analysis is Cumbersome: Manually sifting through large datasets to find tree distribution patterns or recommendations is time-consuming and inefficient.

Solution
The "Tree Intelligence Assistant" provides a multi-faceted solution:
"Recommend Trees by Location" Mode: Offers data-driven suggestions for tree species that are likely to thrive in a specified area, taking into account factors such as latitude, longitude, diameter, native status, city, and state.
"Find Locations for a Tree" Mode: Quickly identifies and lists the most common cities and states where a chosen tree species is found, aiding in research or planning.
"Identify Tree from Image" Mode: Leverages advanced deep learning to instantly predict a tree's species from an uploaded image, providing confidence scores and then linking to its common locations.
This integrated approach makes complex tree-related information accessible and actionable for a wide range of users.

Conclusion
The "Tree Intelligence Assistant" successfully demonstrates the power of machine learning (Nearest Neighbors and CNNs) combined with a user-friendly interface (Streamlit) to solve real-world problems in arboriculture and environmental management.
The application provides valuable insights for tree identification, location-based recommendations, and understanding species distribution.
The robust error handling ensures a more stable and reliable user experience.

Future Enhancements:
Integration with interactive maps (e.g., Folium, Plotly Express) to visualize tree locations.
Incorporation of additional environmental data (e.g., soil type, climate data, precipitation) for more refined recommendations.
Implementation of a feedback mechanism to allow users to contribute to model improvement.
Expansion of the dataset to include more tree species and geographical regions.
Deployment to a cloud platform for wider accessibility.
