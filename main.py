import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

st.title("Advanced Data & Machine Learning Dashboard")

# Sidebar for file upload
st.sidebar.subheader("File Upload")
file_type = st.sidebar.selectbox("Select File Type", ["CSV", "SQL"])

uploaded_file = st.sidebar.file_uploader("Upload your file", type=["csv", "sql"])

if uploaded_file is not None:
    if file_type == "CSV":
        df = pd.read_csv(uploaded_file)
    elif file_type == "SQL":
        conn = sqlite3.connect(uploaded_file)
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table = st.sidebar.selectbox("Select Table", table_names['name'].tolist())
        df = pd.read_sql(f"SELECT * FROM {table}", conn)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    # Advanced filtering
    columns = df.columns.tolist()
    multi_filter_column = st.sidebar.multiselect("Select columns to filter by", columns)
    if multi_filter_column:
        filter_conditions = {}
        for col in multi_filter_column:
            unique_values = df[col].unique()
            selected_values = st.sidebar.multiselect(f"Select values for {col}", unique_values)
            if selected_values:
                filter_conditions[col] = selected_values
        
        # Apply filtering
        for col, values in filter_conditions.items():
            df = df[df[col].isin(values)]
        st.write(df)

    # Convert categorical variables to numerical using one-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    st.sidebar.subheader("Model Selection")
    model_type = st.sidebar.selectbox("Choose Model", [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "K-Means Clustering"
    ])

    # Determine problem type
    target_column = st.sidebar.selectbox("Select target column", df_encoded.columns)
    y = df_encoded[target_column]
    X = df_encoded.drop(target_column, axis=1)

    # Check if the target is continuous or categorical
    is_classification = y.dtype == 'int64' or y.nunique() < 20  # Adjust as necessary for your use case

    # Split data
    test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model training and evaluation
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R^2 Score: {model.score(X_test, y_test):.2f}")

    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier() if is_classification else DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        if is_classification:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        else:
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R^2 Score: {model.score(X_test, y_test):.2f}")
        
        if isinstance(model, DecisionTreeClassifier) or isinstance(model, DecisionTreeRegressor):
            st.subheader("Decision Tree Visualization")
            feature_names = X.columns.tolist()
            fig, ax = plt.subplots(figsize=(15, 10))
            plot_tree(model, feature_names=feature_names, class_names=str(model.classes_) if is_classification else None, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Decision Tree visualization is not supported for the selected model.")

    elif model_type == "Random Forest":
        model = RandomForestClassifier() if is_classification else RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        if is_classification:
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        else:
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R^2 Score: {model.score(X_test, y_test):.2f}")
        
        if is_classification:
            st.subheader("Feature Importance")
            feature_importances = pd.Series(model.feature_importances_, index=X.columns)
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.sort_values().plot(kind='barh', ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

    elif model_type == "Support Vector Machine":
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif model_type == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif model_type == "Naive Bayes":
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    elif model_type == "K-Means Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        model = KMeans(n_clusters=n_clusters)
        y_pred = model.fit_predict(X)
        st.subheader("K-Means Clustering Results")
        df['Cluster'] = y_pred
        st.write(df.head())

    # Data Visualization
    st.sidebar.subheader("Data Visualization")
    plot_type = st.sidebar.selectbox("Select Plot Type", ["None", "Histogram", "Box Plot", "Scatter Plot"])

    if plot_type == "Histogram":
        st.subheader("Histogram")
        column = st.sidebar.selectbox("Select column for histogram", df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        df[column].hist(bins=30, edgecolor='k', ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        st.subheader("Box Plot")
        column = st.sidebar.selectbox("Select column for box plot", df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Box Plot of {column}")
        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        x_column = st.sidebar.selectbox("Select x-axis column", df.columns)
        y_column = st.sidebar.selectbox("Select y-axis column", df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[x_column], df[y_column], alpha=0.5)
        ax.set_title(f"Scatter Plot of {x_column} vs {y_column}")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        st.pyplot(fig)
