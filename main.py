import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np
import sqlite3

# Function to plot the decision tree
def plot_decision_tree(model, X):
    fig, ax = plt.subplots(figsize=(20, 10))  # Set figure size
    feature_names = list(X.columns)  # Get feature names
    plot_tree(model, feature_names=feature_names, filled=True, ax=ax)  # Plot the tree
    st.pyplot(fig)  # Display the plot in Streamlit

# File uploader for CSV and SQL files
uploaded_file = st.file_uploader("Choose a CSV or SQL file", type=["csv", "sql"])

if uploaded_file is not None:
    # Handle CSV file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)  # Load CSV into DataFrame
    # Handle SQL file
    elif uploaded_file.name.endswith('.sql'):
        conn = sqlite3.connect(':memory:')  # Create an in-memory SQLite database
        with open(uploaded_file.name, 'r') as f:
            sql_query = f.read()  # Read SQL query
        df = pd.read_sql(sql_query, conn)  # Execute SQL query and load into DataFrame

    st.subheader("Data Preview")
    st.write(df.head())  # Display the first few rows of the DataFrame

    st.subheader("Data Summary")
    st.write(df.describe(include='all'))  # Display summary statistics of the DataFrame

    st.subheader("Filter Data")
    columns = df.columns.tolist()  # List of DataFrame columns
    selected_column = st.selectbox("Select column to filter by", columns)  # Select column for filtering
    unique_values = df[selected_column].unique()  # Get unique values of the selected column
    selected_value = st.selectbox("Select value", unique_values)  # Select value to filter by
    filtered_df = df[df[selected_column] == selected_value]  # Filter the DataFrame
    st.write(filtered_df)  # Display the filtered DataFrame

    st.subheader("Select Machine Learning Task")
    task = st.selectbox("Choose a task", [
        "Linear Regression",
        "Logistic Regression",
        "Decision Trees",
        "Random Forests",
        "Support Vector Machines (SVM)",
        "K-Nearest Neighbors (KNN)",
        "Naive Bayes",
        "K-Means Clustering",
        "Hierarchical Clustering",
        "Principal Component Analysis (PCA)",
        "Anomaly Detection",
        "Reinforcement Learning",
        "Deep Learning",
        "Natural Language Processing (NLP)",
        "Time Series Analysis",
        "Dimensionality Reduction",
        "Cross-Validation",
        "Hyperparameter Tuning"
    ])

    if task in ["Linear Regression", "Logistic Regression", "Decision Trees", "Random Forests", "Support Vector Machines (SVM)", "K-Nearest Neighbors (KNN)", "Naive Bayes"]:
        st.subheader("Select Target Column")
        target_column = st.selectbox("Target Column", columns)  # Select target column

        # Determine if the target column is for classification or regression
        if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
            is_classification = True
        else:
            is_classification = False

        X = df.drop(target_column, axis=1)  # Features
        y = df[target_column]  # Target

        # Handle categorical features
        X = pd.get_dummies(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        # Select and train the model based on the task
        if task == "Linear Regression" and not is_classification:
            model = LinearRegression()
        elif task == "Logistic Regression" and is_classification:
            model = LogisticRegression(max_iter=1000)
        elif task == "Decision Trees" and is_classification:
            model = DecisionTreeClassifier()
        elif task == "Random Forests" and is_classification:
            model = RandomForestClassifier()
        elif task == "Support Vector Machines (SVM)" and is_classification:
            model = SVC()
        elif task == "K-Nearest Neighbors (KNN)" and is_classification:
            model = KNeighborsClassifier()
        elif task == "Naive Bayes" and is_classification:
            model = GaussianNB()
        else:
            st.error("The selected task is not suitable for the type of data.")
            st.stop()

        try:
            model.fit(X_train, y_train)  # Train the model
            predictions = model.predict(X_test)  # Make predictions

            if is_classification:
                st.subheader(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
                st.write(classification_report(y_test, predictions))  # Display classification report
                if task == "Decision Trees":
                    st.subheader("Decision Tree Visualization")
                    plot_decision_tree(model, X)  # Plot decision tree
            else:
                st.subheader(f"Mean Squared Error: {mean_squared_error(y_test, predictions):.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif task == "K-Means Clustering":
        st.subheader("Select Number of Clusters")
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)  # Select number of clusters
        X = df.dropna()  # Drop missing values
        X = pd.get_dummies(X)  # Handle categorical features
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)  # Fit K-Means model
        df['Cluster'] = kmeans.labels_  # Assign cluster labels
        st.write(df.head())  # Display DataFrame with clusters
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='Cluster', data=df, ax=ax)  # Scatter plot of clusters
        st.pyplot(fig)

    elif task == "Hierarchical Clustering":
        from scipy.cluster.hierarchy import dendrogram, linkage
        st.subheader("Hierarchical Clustering")
        X = df.dropna()  # Drop missing values
        X = pd.get_dummies(X)  # Handle categorical features
        linked = linkage(X, 'single')  # Perform hierarchical clustering
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', ax=ax)  # Plot dendrogram
        st.pyplot(fig)

    elif task == "Principal Component Analysis (PCA)":
        st.subheader("PCA Visualization")
        X = df.dropna()  # Drop missing values
        X = pd.get_dummies(X)  # Handle categorical features
        pca = PCA(n_components=2)  # Initialize PCA
        principalComponents = pca.fit_transform(X)  # Perform PCA
        pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])  # Create DataFrame for PCA components
        st.write(pca_df.head())  # Display PCA DataFrame
        fig, ax = plt.subplots()
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, ax=ax)  # Scatter plot of PCA components
        st.pyplot(fig)

    elif task == "Anomaly Detection":
        from sklearn.ensemble import IsolationForest
        st.subheader("Anomaly Detection")
        X = df.dropna()  # Drop missing values
        X = pd.get_dummies(X)  # Handle categorical features
        model = IsolationForest()  # Initialize Isolation Forest
        anomalies = model.fit_predict(X)  # Detect anomalies
        df['Anomaly'] = anomalies  # Add anomaly column
        st.write(df.head())  # Display DataFrame with anomalies
        st.subheader("Anomaly Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='Anomaly', data=df, ax=ax)  # Scatter plot of anomalies
        st.pyplot(fig)

    elif task == "Reinforcement Learning":
        st.subheader("Reinforcement Learning")
        # Simplified example placeholder
        st.write("This is a placeholder for reinforcement learning.")

    elif task == "Deep Learning":
        st.subheader("Deep Learning")
        # Simplified example placeholder
        st.write("This is a placeholder for deep learning.")

    elif task == "Natural Language Processing (NLP)":
        st.subheader("NLP Example")
        text_data = st.text_area("Enter text data for NLP")
        if text_data:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()  # Initialize CountVectorizer
            X = vectorizer.fit_transform([text_data])  # Vectorize text data
            st.write("Feature names:", vectorizer.get_feature_names_out())  # Display feature names
            st.write("Vectorized data:", X.toarray())  # Display vectorized data

    elif task == "Time Series Analysis":
        st.subheader("Time Series Analysis")
        # Simplified example placeholder
        st.write("This is a placeholder for time series analysis.")

    elif task == "Dimensionality Reduction":
        st.subheader("Dimensionality Reduction Example")
        # Simplified example placeholder
        st.write("This is a placeholder for dimensionality reduction.")

    elif task == "Cross-Validation":
        st.subheader("Cross-Validation Example")
        # Simplified example placeholder
        st.write("This is a placeholder for cross-validation.")

    elif task == "Hyperparameter Tuning":
        st.subheader("Hyperparameter Tuning Example")
        # Simplified example placeholder
        st.write("This is a placeholder for hyperparameter tuning.")

    else:
        st.error("Selected task is not supported yet.")
