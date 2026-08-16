import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

from validation import validate_dataframe
from visualizations import correlation_heatmap, scatter_plot, histogram_plot, bar_plot, box_plot
from analysis import analyze_missing_values, analyze_duplicates, analyze_column_types, numerical_analysis, outlier_detection, categorical_analysis, correlation_analysis

from preprocessing import prepare_data
from ml_models import train_classification_model,train_regression_model,evaluate_classification_model, evaluate_regression_model, get_feature_importance

from io import BytesIO
import io

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_sample_csv(path):
    return pd.read_csv(path)

@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

# Page Configuration
st.set_page_config(page_title="Dataset Explorer & Analyzer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

def load_css():
    with open("style.css","r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

load_css()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin:0; font-size:24px;">
            📊 DataLens
        </h2>
        <p style="color:#718096; margin-top:5px;">
            Interactive Data Science Platform
        </p>
    </div>""", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 🧭 Navigation")

    page = st.radio("Navigation", ["Overview", "Analysis", "Visualizations", "ML Models", "About","Instructions", "Help", "Contact"], label_visibility="collapsed")
    st.divider()

    st.markdown("### 📁 Dataset")

    # File Uploader
    uploaded_file= st.file_uploader("Choose a File", type=['csv','xlsx'])

    # Sample data
    Sample_df = {
    "🏨 Hotel Booking — Classification + Regression":
        "sample_df/hotel_bookings.csv",

    "👥 Credit Card — Classification":
        "sample_df/creditcard.csv",

    "🛒 Dirty Retail Sales — Data Cleaning":
        "sample_df/retail_store_sales.csv",

    "🏪 Retail Analysis — Business Analytics":
        "sample_df/new_retail_data.csv",

    "🎬 Netflix — EDA & Visualization":
        "sample_df/netflix_titles.csv",}

    st.markdown("### 🧪 Try Sample Data")

    sample_options = ["None"] + list(Sample_df.keys())

    sample_dataset = st.selectbox(
        "Choose a sample dataset",
        sample_options,
        key="sample_dataset")
    if sample_dataset != None:
        if st.button("📥 Load Sample Dataset", width="stretch"):
            sample_path = Sample_df[sample_dataset]
            try:
                if not os.path.exists(sample_path):
                    st.error(f"Sample dataset not found: `{sample_path}`")
                else:
                    with st.spinner("Loading sample dataset..."):

                        df = load_sample_csv(sample_path)

                        # Validate dataset
                        valid, message = validate_dataframe(df)

                        if not valid:
                            st.error(f"❌ Dataset validation failed: {message}")
                        else:

                            # Store original dataset
                            st.session_state["original_df"] = df.copy()

                            # Store working dataset
                            st.session_state["processed_df"] = df.copy()

                            # Store dataset information
                            st.session_state["current_file"] = sample_path
                            st.session_state["current_dataset_name"] = sample_dataset

                            st.success(f"✅ {sample_dataset} loaded successfully!")
                            st.rerun()
            except pd.errors.EmptyDataError:
                st.error("❌ The sample dataset is empty.")

            except pd.errors.ParserError:
                st.error("❌ Unable to parse the sample CSV file.")

            except Exception as e:
                st.error(f"❌ Unable to load sample dataset: {e}")        

# Main Title
st.title("📊 DataLens")
st.caption("Upload, explore, analyze, visualize and model your data — all in one place.")

# Load Dataset
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df= load_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df= load_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            df= None

        if df is not None:
            # Validate uploaded datset
            valid, message = validate_dataframe(df)
            if not valid:
                st.error(message)
                st.stop()

            st.success("File uploaded and validated successfully!")

            #store original and processed datasets
            file_id = uploaded_file.name
            if("current_file" not in st.session_state or st.session_state["current_file"] != file_id):
                st.session_state["current_file"] = file_id

                # keep original dataset unchanged
                st.session_state["original_df"] = df.copy()

                # Working dataset that can be cleaned
                st.session_state["processed_df"] = df.copy()

            # Use processed dataset throughout the application 
            df = st.session_state["processed_df"]
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

elif "processed_df" in st.session_state:
    # Sample dataset or already processed dataset
    df = st.session_state["processed_df"]

else:
    df =None

# Stop pages from running without a dataset
if df is None:
    st.info("👈 Please upload a CSV/Excel file or load a sample dataset from the sidebar.")
    st.stop()

# Pages
# Overview Page
if page == "Overview":

    st.markdown("## 📊 Dataset Overview")
    st.caption("A quick summary of the uploaded dataset.")

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{total_rows:,}")

    with col2:
        st.metric("Total Columns", f"{total_columns:,}")

    with col3:
        st.metric("Missing Values", f"{missing_values:,}")

    with col4:
        st.metric("Duplicates Rows", f"{duplicate_rows:,}")
                
    # --- Dataset Preview ----
    st.markdown("## 👀 Dataset Preview")
    st.caption("First 5 rows of the uploaded dataset.")
    st.dataframe(df.head(), width="stretch",hide_index=True)

    # ---- Data Information ----
    with st.expander("📋 Dataset Information", expanded=False):

        buffer = io.StringIO()
        df.info(buf=buffer)

        st.code(buffer.getvalue(),language="text")


# Analysis Page
elif page == "Analysis":
    st.markdown("""<div class="analysis-section">
                                    <h1>🔍 Data Analysis</h1>
                                    <p class="analysis-description">
                                     Explore data quality, statistical patterns, categorical distributions,
                                outliers and relationships.
                                    </p>
                                    </div>""",unsafe_allow_html=True)

    analysis_type = st.selectbox("🔍 What would you like to analyze?", ["Data Quality", "Statistical Analysis", "Categorical Analysis", "Outlier Analysis", "Correlation Analysis"],key="analysis_type")

    if analysis_type == "Data Quality":
        st.markdown("### 🧹 Data Quality")
        st.caption("Check and improve the quality of your dataset.")

        # Reset Dataset
        col1, col2 = st.columns([3,1])

        with col1:
            st.write("Reset all cleaning changes and restore the original dataset.")
        with col2:
            if st.button("🔄 Reset Dataset",key="reset_dataset"):
                st.session_state["processed_df"] = (st.session_state["original_df"].copy())
                st.success("✅ Dataset restored to the original version.")
                st.rerun()
                # Current dataset status
                st.info(f"Current dataset: {len(df):,} rows × {len(df.columns):,} columns") 
                    
        # --- Missing Values Analysis ---
        with st.expander("🧹 Missing Values", expanded= True):
            try:
                missing_result = analyze_missing_values(df)

                if missing_result["has_missing_values"]:
                    st.markdown("### Missing Value Summary")
                    st.dataframe(missing_result["missing_info_df"],width="stretch",hide_index=True)

                    # Columns containing missing values
                    missing_info_df = missing_result["missing_info_df"]
                    missing_columns = (missing_info_df[missing_info_df["Missing Count"]>0].index.tolist())
                    if missing_columns:
                        st.markdown("#### 🛠️ Handle Missing Values")
                        selected_column = st.selectbox("Select a column: ", missing_columns,key="missing_value_column")

                        # Determine column type
                        if pd.api.types.is_numeric_dtype(df[selected_column]):
                            action = st.selectbox("Choose an action: ", ["Remove rows", "Impute with Mean", "Impute with Median"], key="missing_value_action")
                        else:    
                            action = st.selectbox("Choose an action: ", ["Remove rows", "Impute with Mode"], key="missing_value_action")

                        if st.button("Apply Changes", key="apply_missing_changes"):
                            if action == "Remove rows":
                                df = df.dropna(subset = [selected_column])
                            elif action == "Impute with Mean":
                                df[selected_column] = df[selected_column].fillna(df[selected_column].mean())
                            elif action == "Impute with Median":
                                df[selected_column] = df[selected_column].fillna(df[selected_column].median())
                            elif action == "Impute with Mode":
                                mode_value = (df[selected_column].mode())
                                if not mode_value.empty:
                                    df[selected_column] =  df[selected_column].fillna(mode_value.iloc[0])

                            # Save updated Dataset
                            st.session_state["processed_df"] = df.copy()
                            st.success(f"✅ '{selected_column}' updated successfully.")
                            st.rerun()
                else:
                    st.success("No missing values detected.")

            except Exception as e:
                st.error(f"Unable to analyze missing values: {e}")

        # --- Duplicate Analysis ---
        with st.expander("🔁 Duplicate Analysis",expanded=False):
            try:
                duplicate_result = analyze_duplicates(df)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Duplicate Rows",duplicate_result["duplicate_count"])

                with col2:
                    st.metric("Duplicate Percentage", f"{duplicate_result['duplicate_percentage']:.2f}%")

                if duplicate_result["has_duplicates"]:
                    st.warning("Duplicate rows detected in the dataset.")
                    st.markdown("#### Duplicate Rows")
                    st.dataframe(duplicate_result["duplicate_rows_df"],width="stretch",hide_index=True)

                    st.markdown("#### 🛠️ Handle Duplicates")
                    if st.button("🗑️ Remove Duplicate Rows",key="remove_duplicates"):
                        rows_before = len(df)
                        df = df.drop_duplicates().reset_index(drop = True)
                        rows_after= len(df)

                        # Save updated dataset
                        st.session_state["processed_df"] = df.copy()
                        removed_rows = rows_before - rows_after
                        st.success(f"✅ {removed_rows} duplicate rows removed successfully.")
                        st.rerun()
                else: 
                    st.success("No duplicate rows detected.")

            except Exception as e:
                st.error(f"Unable to analyze duplicate rows: {e}")

        # --- Column Type Analysis ---
        with st.expander("🔤 Column Type Analysis",expanded=False):
            try:

                column_types = analyze_column_types(df)

                type_df = pd.DataFrame(list(column_types.items()), columns=["Column", "Detected Type"])
                st.markdown("#### 📋 Column Type Overview")
                st.dataframe(type_df, width="stretch", hide_index=True)

                # Datatype Conversion
                st.markdown("#### 🔄 Convert Column Datatype")
                selected_column = st.selectbox("Select a column:",df.columns.tolist(),key="datatype_column")
                current_type = str(df[selected_column].dtype)
                st.write( f"**Current datatype:** `{current_type}`")
                new_type = st.selectbox("Convert to:", ["Numeric","String","Category", "Datetime","Boolean"],key="datatype_conversion")

                if st.button("🔄 Apply Conversion", key="apply_datatype_conversion"):
                    try:
                        if new_type == "Numeric":
                            df[selected_column] = pd.to_numeric(df[selected_column],errors="raise")

                        elif new_type == "String":
                            df[selected_column] = (df[selected_column].astype("string"))

                        elif new_type == "Category":
                            df[selected_column] = (df[selected_column].astype("category"))

                        elif new_type == "Datetime":
                            df[selected_column] = pd.to_datetime(df[selected_column],errors="raise")

                        elif new_type == "Boolean":
                            df[selected_column] = (df[selected_column].astype("boolean"))

                        # Save converted dataset
                        st.session_state["processed_df"] = df.copy()
                        st.success(f"✅ '{selected_column}' converted to {new_type} successfully.")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Unable to convert '{selected_column}' "
                                            f"to {new_type}: {e}")
                                    
            except Exception as e:
                st.error(f"Unable to analyze column type: {e}")

    # STATISTICAL ANALYSIS
    elif analysis_type == "Statistical Analysis":
        st.markdown("### 📈 Statistical Analysis")
        st.caption("Explore descriptive statistics and numerical characteristics of your data.")
                
        try:
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if numerical_columns:
                # Select numerical column
                selected_column = st.selectbox("Select a numerical column:",numerical_columns,key="statistical_column")

                # Selected column data
                column_data = df[selected_column].dropna()

                # Descriptive statistics
                count = column_data.count()
                mean = column_data.mean()
                median = column_data.median()
                std = column_data.std()
                minimum = column_data.min()
                maximum = column_data.max()
                q1 = column_data.quantile(0.25)
                q3 = column_data.quantile(0.75)
                iqr = q3 - q1

                # Display statistics
                st.markdown(f"#### 📊 Statistics for `{selected_column}`")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Count", f"{count:,}")

                with col2:
                    st.metric("Mean", f"{mean:.2f}")

                with col3:
                    st.metric("Median", f"{median:.2f}")

                with col4:
                    st.metric("Std. Deviation", f"{std:.2f}")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Minimum", f"{minimum:.2f}")

                with col2:
                    st.metric("Q1 (25%)", f"{q1:.2f}")

                with col3:
                    st.metric("Q3 (75%)", f"{q3:.2f}")

                with col4:
                    st.metric("Maximum", f"{maximum:.2f}")

                st.markdown("#### 📐 Interquartile Range")
                st.metric("IQR",f"{iqr:.2f}")   

                # Distribution Statistics
                st.markdown("#### 📊 Distribution Statistics")

                skewness = column_data.skew()
                kurtosis = column_data.kurt()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Skewness", f"{skewness:.4f}")

                with col2:
                    st.metric("Kurtosis", f"{kurtosis:.4f}")
                # Percentile Analysis
                st.markdown("#### 📌 Percentile Analysis")
                percentile = st.selectbox("Select percentile:", [10,25,50,75,90,95,99],key="statistical_percentile")

                percentile_value = column_data.quantile(percentile/100)
                st.metric(f"{percentile}th Percentile", f"{percentile_value:.2f}")
            else:
                st.info("No numerical columns found in the dataset.")                             
        except Exception as e:
            st.error(f"Unable to analyze numerical type: {e}")

    # --- Categorical Analysis ---
    elif analysis_type == "Categorical Analysis":
        st.markdown("### 🏷️ Categorical Analysis")
        st.caption("Explore the distribution and frequency of categorical variables.")
        try:
            # get categorical column
            categorical_columns = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
            if categorical_columns:
                selected_column = st.selectbox("Select a categorical column:", categorical_columns,key="categorical_analysis_column")

                # Remove missing values only for analysis
                column_data = df[selected_column].dropna()

                # Basic stats
                unique_count= column_data.nunique()
                category_counts = (column_data.value_counts())
                category_percentages = (column_data.value_counts(normalize=True) * 100)

                most_frequent = category_counts.index[0]
                most_frequent_count = category_counts.iloc[0]
                most_frequent_percentage = (category_percentages.iloc[0])

                least_frequent = category_counts.index[-1]
                least_frequent_count = category_counts.iloc[-1]
                least_frequent_percentage = (category_percentages.iloc[-1])

                # Summary
                st.markdown(f"#### 📊 Analysis for `{selected_column}`")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Unique Categories",f"{unique_count:,}")

                with col2:
                    st.metric("Most Frequent",str(most_frequent))

                with col3:
                    st.metric("Frequency",f"{most_frequent_count:,}")

                with col4:
                    st.metric("Percentage",f"{most_frequent_percentage:.2f}%")

                # Least Frequent Category
                st.markdown("#### 📉 Least Frequent Category")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Category",str(least_frequent))

                with col2:
                    st.metric("Frequency",f"{least_frequent_count:,}")

                with col3:
                    st.metric("Percentage", f"{least_frequent_percentage:.2f}%")

                # Category Distribution
                st.markdown("#### 📋 Category Distribution")

                # Category distribution
                category_df = pd.DataFrame({
                        "Category": category_counts.index.astype(str),
                        "Count": category_counts.values,
                        "Percentage": category_percentages.values.round(2)
                })

                st.dataframe(category_df, width="stretch", hide_index=True)

                # Top-N Categories
                st.markdown("#### 🔝 Top-N Categories")
                top_n = st.selectbox("Show Top: ", [5,10,15,20],index=1,key="categorical_top_n")
                top_categories = (category_counts.head(top_n))
                top_percentages = (category_percentages.head(top_n))

                top_category_df = pd.DataFrame({
                    "Category" : top_categories.index.astype(str),
                    "Count": top_categories.values,
                    "Percentage": top_percentages.values.round(2)
                })

                st.write(f"Showing the top {top_n} most frequent categories.")
                st.dataframe(top_category_df, width="stretch", hide_index=True)

            else:
                st.info("No Categorical columns found.")    
        except Exception as e:
            st.error(f"Unable to analyze categorical type: {e}")

    # Outlier Detection
    elif analysis_type == "Outlier Analysis":
        st.markdown("### 📦 Outlier Analysis")
        st.caption("Identify potential outliers in numerical columns using the existing outlier detection method.")
                    
        try:
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

            # Exclude binary numerical columns (0/1)
            numerical_columns = [col for col in numerical_columns if df[col].dropna().nunique() > 2]
            if not numerical_columns:
                st.info("No suitable numerical columns found for outlier analysis.")

            else:
                # Select numerical column
                selected_column = st.selectbox("Select a numerical column: ",numerical_columns,key="outlier_analysis_column")

                # Remove missing values only for analysis
                column_data = df[selected_column].dropna()
                if column_data.empty:
                    st.info("No valid numerical values available for this column.")

                else:
                    # IQR
                    q1 = column_data.quantile(0.25)
                    q3 = column_data.quantile(0.75)

                    iqr = q3 - q1

                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)

                    # Detect outlier
                    outlier_mask= ( (column_data < lower_bound) | (column_data > upper_bound))
                    outlier_indices = column_data.index[outlier_mask]
                    outlier_count = len(outlier_indices)
                    total_values = len(column_data)

                    outlier_percentage = (outlier_count/total_values) *100

                    # summary metrics
                    st.markdown(f"#### 🔎 Outlier Summary for `{selected_column}`") 

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Values", f"{total_values:,}")
                                
                    with col2:
                        st.metric("Potential Outliers",f"{outlier_count:,}")
                                
                    with col3:
                        st.metric("Outlier Percentage",f"{outlier_percentage:.2f}%")

                    # IQR Statistics
                    st.markdown("#### 📐 IQR Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Q1 (25%)", f"{q1:.2f}")

                    with col2:
                        st.metric("Q3 (75%)", f"{q3:.2f}")

                    with col3:
                        st.metric("IQR", f"{iqr:.2f}")

                    with col4:
                        st.metric("Lower Bound", f"{lower_bound:.2f}")

                    st.metric("Upper Bound",f"{upper_bound:.2f}")

                    # Interpretation
                    st.markdown("#### 💡 Outlier Detection Rule")
                    st.info(
                                    f"Values below **{lower_bound:.2f}** "
                                    f"or above **{upper_bound:.2f}** "
                                    f"are considered potential outliers using "
                                    f"the 1.5 × IQR rule.")

                    if outlier_count > 0:
                                   
                        # Outlier Handling
                        st.markdown("#### 🛠️ Handle Outliers")
                                    
                        if st.button(f"🗑️ Remove {outlier_count:,} Detected Outliers",key="remove_outliers"):
                            # Remove ALL currently detected
                            # outlier rows at once
                            cleaned_df = df.drop(index= outlier_indices).copy()

                            # Save cleaned dataset
                            st.session_state["processed_df"] = cleaned_df
                            st.success(f"✅ {outlier_count:,} outlier rows removed successfully.")
                            st.rerun()
                    else:
                        st.success( "✅ No potential outliers detected using the current IQR boundaries.")
                            
        except Exception as e:
            st.error(f"Unable to detect outliers: {e}")
    # Correlation Analysis
    elif analysis_type == "Correlation Analysis":
        st.markdown("### 🔗 Correlation Analysis")
        st.caption("Explore relationships between numerical variables.")
        try:
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numerical_columns) >= 2:
                # Select columns for correlation analysis
                selected_column = st.multiselect("Select columns to analyze: ", numerical_columns, default=numerical_columns, key="correlation_columns")
                if len(selected_column) >=2:
                    # Calculate correlation matrix
                    correlation_matrix = df[selected_column].corr()

                    # Display Correlation Matrix
                    st.markdown("#### 📊 Correlation Matrix")
                    st.dataframe(correlation_matrix.round(3), width="stretch")

                    # Correlation Threshold
                    st.markdown("#### 🔍 Strong Relationships")
                    threshold = st.slider("Minimum absolute correlation:",min_value=0.1,max_value=1.0,value=0.7,step=0.1,key="correlation_threshold")

                    # Find strong relationships
                    strong_relationships = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            column_1 = (correlation_matrix.columns[i])
                            column_2 = (correlation_matrix.columns[j])
                            correlation_value = (correlation_matrix.iloc[i,j])

                            if (pd.notna(correlation_value) and abs(correlation_value) >= threshold):
                                if correlation_value > 0:
                                    relationship = "Positive"
                                else:
                                    relationship = "Negative"

                                strong_relationships.append({
                                    "Feature 1": column_1,
                                    "Feature 2": column_2,
                                    "correlation": round(correlation_value,3),
                                    "Relationship":relationship
                                })                            
                            # Display Strong Relationships
                            if strong_relationships:
                                    
                                relationship_df = pd.DataFrame(strong_relationships)
                                st.dataframe(relationship_df, width="stretch",hide_index=True)
                            else:
                                st.info( f"No relationships found with" f"absolute correlation ≥ {threshold:.1f}.")
                                    

                else: 
                    st.info("Please select at least two numerical columns for correlation analysis.")

            else:
                st.info("At least two numerical columns are required for correlation analysis.")

        except Exception as e:
            st.error(f"Unable to analyze correlation: {e}")

    # ---- Export Processed Data 
    st.markdown("## Export Data")
    st.caption("Download the current dataset after validation and processing.")

    try:
        col1, col2= st.columns(2)

        #CSV Export
        with col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(label="Download Data as CSV", data= csv_data, file_name="processed_data.csv",mime="text/csv",width="stretch")

        # Excel Export
        with col2:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer,index=False,sheet_name="Processed Data")

                excel_buffer.seek(0)
                st.download_button(label="Download Excel", data= excel_buffer.getvalue(), file_name="processed_data.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",width="stretch")
    except Exception as e:
        st.error(f"Unable to export processed data: {e}")

# --- Visualization Page ---
elif page == "Visualizations":
    st.markdown("""
                 <div class="visualization-section">
                <h1>📊 Data Visualizations</h1>
                <p class="visualization-description">
                Explore distributions, relationships and patterns through interactive visual analysis.
                </p>
                </div>""",unsafe_allow_html=True)

    st.markdown("### 🎨 Choose Visualization")
    visualization_type = st.selectbox( "Visualization Type",["Histogram", "Bar Chart", "Box Plot", "Scatter Plot", "Heatmap"],key="visualization_type") 
                

    # ---- Histogram ----
    if visualization_type == "Histogram":
        numerical_columns= df.select_dtypes(include=np.number).columns.tolist()

        if numerical_columns:
            selected_column = st.selectbox("Select a numerical column:", numerical_columns, key="histogram_column")

            try:
                fig = histogram_plot(df,selected_column)
                if fig is not None:
                    st.pyplot(fig,width="content")
                    plt.close(fig)

                else:
                    st.warning("Unable to create histogram for the selected column.")
            except Exception as e:
                st.error(f"Unable to create histogram: {e}")
        else:
            st.warning("No numerical columns available.")

    elif visualization_type == "Bar Chart":
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if categorical_columns:
            selected_column = st.selectbox("Select a categorical column: ",categorical_columns,key="bar_column")

            try:                    
                fig = bar_plot(df,selected_column)
                if fig is not None:
                    st.pyplot(fig,width="content")
                    plt.close(fig)

                else:
                    st.warning("Unable to create bar chart for the selected column.")

            except Exception as e:
                st.error(f"Unable to create barplot: {e}")
        else:
            st.warning("No categorical columns available.")

    elif visualization_type == "Box Plot":
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

        if numerical_columns:
            selected_column = st.selectbox("Select a numerical column:", numerical_columns, key="box_visualization_column")

            try:
                fig = box_plot(df, selected_column)

                if fig is not None:
                    st.pyplot(fig,width="content")
                    plt.close(fig)
                else:
                    st.warning("Unable to create bar chart for the selected column.")
            except Exception as e:
                st.error(f"Unable to create boxplot: {e}")

        else:
            st.warning("No numerical columns available.")

    elif visualization_type == "Scatter Plot":
        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(numerical_columns) >= 2:

            st.markdown("### 🔗 Select Variables")

            col1, col2 = st.columns(2)

            with col1:
                x_column = st.selectbox("X-axis", numerical_columns, key= "scatter_x")

            with col2:
                y_column = st.selectbox("Y-axis", numerical_columns,key= "scatter_y")

            try:
                fig = scatter_plot(df, x_column, y_column)
                if fig is not None:
                    st.pyplot(fig,width="content")
                    plt.close(fig)

                else:
                    st.warning("Unable to create scatter plot with the selected columns.")
            except Exception as e:
                st.error(f"Unable to create scatterplot: {e}")

        else:
            st.warning("At least two numerical columns are required for a scatter plot.")

    elif visualization_type == "Heatmap":
        st.markdown("### 🔥 Correlation Heatmap")
        try:
            fig = correlation_heatmap(df)

            if fig is not None:
                st.pyplot(fig,width="content")
                plt.close(fig)

            else:
                st.warning("At least two numerical columns are required for a heatmap.")
        except Exception as e:
            st.error(f"Unable to create heatmap: {e}")

# ML Models Page
elif page == "ML Models":
    st.markdown("## 🤖 Machine Learning Models")
    st.caption("Train, compare and evaluate machine learning models on your uploaded dataset.")

    st.write("Build and evaluate machine learning models using your uploaded dataset.")

    # Target column selection
    st.markdown("### 🎯 Target Selection")
    st.caption("Choose the column you want the model to predict.")
    target_column = st.selectbox("Target column: ", df.columns.tolist(),key= "ml_target")

    st.success(f"Target column selected: {target_column}")

    # Problem Type Detection
    st.markdown("### Problem Type")
    problem_type_mode = st.radio("How should DataLens determine the problem type?",["Automatic Detection","Classification","Regression"], horizontal=True, key="problem_type_mode")

    target_data = df[target_column]

    # Automatic detction
    if problem_type_mode == "Automatic Detection":
                    
        if pd.api.types.is_numeric_dtype(target_data):
            unique_values = target_data.nunique()
            if unique_values <= 10:
                problem_type = "Classification"
            else:
                problem_type = "Regression"

        else:
            problem_type = "Classification"

    elif problem_type_mode == "Classification":
        problem_type = "Classification"

    else:
        problem_type = "Regression"

    st.info(f"Selected Problem Type: **{problem_type}**")

    # Display Problem Type
    st.markdown("### 🧠 Problem Detection")

    col1, col2 = st.columns(2)

    with col1:
        if problem_type == "Classification":
            st.success(f"🎯 Problem Type: {problem_type}")

        else:
            st.info(f"📈 Problem Type: {problem_type}")

    with col2:
        st.metric("Target Unique Values", target_data.nunique())

    # Target Information
    with st.expander("📋 Target Information", expanded=False):
        target_info = pd.DataFrame({
        "Property": ["Column Name", "Data Type", "Unique Values", "Missing Values", "Total Values"],
        "Value" : [str(target_column), str(target_data.dtype), str(target_data.nunique()), str(target_data.isnull().sum()), str(target_data.count())]
        })

        st.dataframe(target_info, width="stretch", hide_index=True)

    # Feature Selection
    st.markdown("### 🔍 Feature Selection")
    st.caption("Select the features that will be used for training.")
    feature_columns = [column for column in df.columns if column != target_column]

    st.write("The following columns are availble as potential features:")

    selected_features = st.multiselect("Input Features:", feature_columns, default=feature_columns, key="ml_features")

    if selected_features:
        st.success(f"{len(selected_features)} feature(s) selected.")

        #Feature summary
        with st.expander("📊 Selected Feature Summary",expanded=False):
            st.subheader("Selected Feature Summary")

            feature_df = df[selected_features]
            feature_summary = pd.DataFrame({
                "Feature": selected_features, "Data Type": [str(feature_df[column].dtype) for column in selected_features],
                "Unique Values": [feature_df[column].nunique() for column in selected_features],
                "Missing Values": [feature_df[column].isnull().sum() for column in selected_features]
                })

            st.dataframe(feature_summary,width="stretch",hide_index=True)
    else:
        st.warning("Please select at least one feature.")

    # Training Configuration
    st.markdown("### ⚙️ Training Configuration")
    st.caption("Configure the training and model evaluation settings.")
    test_size = st.slider("Test Size", min_value=0.1, max_value=0.4,value=0.2,step=0.05)

    st.write(f"Training data: **{100 - test_size}%**  |  "f"Testing data: **{test_size}%**")

    # Model Selection
    st.markdown("### Model Selection")
    st.caption("Choose whether to compare all models or train a specific model.")
    training_mode = st.radio("Training Mode", ["Compare All Models", "Train Selected Model"], horizontal=True,key="training_mode")

    # Available models based on problem type
    if problem_type == "Classification":
        model_options = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    else:
        model_options = ["Linear Regression","Random Forest", "Gradient Boosting"]

    # Show model selection only when required
    if training_mode == "Train Selected Model":
        selected_model = st.selectbox("Select Model",model_options,key="selected_model")

    # Prepare Data Button
    st.markdown("### 🛠️ Data Preparation")

    if st.button("Prepare Data", key="prepare_data_button",width="stretch"):
        if not selected_features:
            st.warning("Please select at least one Feature.")
        else:
            try:
                with st.spinner("Preparing Data..."):

                    result = prepare_data(df, target_column, selected_features,test_size=test_size, problem_type=problem_type)

                    st.session_state["ml_data"] = result
                st.success("Data Prepared Successfully!")
            except Exception as e:
                st.error(f"Error while preparing data: {e}")

    # Prepare data results
    if "ml_data" in st.session_state:
        ml_data = st.session_state["ml_data"]

        st.markdown("### 📦 Prepared Dataset")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Training Samples", len(ml_data["y_train"]))

        with col2:
            st.metric("Testing Samples", len(ml_data["y_test"]))

        with col3:
            st.metric("Features", len(ml_data["feature_names"]))

        st.success("Preprocessing completed. Data is ready for model training.")    
                        

    # Model Training
    st.markdown("### 🚀 Model Training")
    st.caption("Train multiple machine learning models " "and compare their performance.")

    if st.button("🚀 Train Models", key="train_models_button",width="stretch"):

        if "ml_data" not in st.session_state:
            st.warning("Please prepare the data before training models.")

        else:
            try:
                with st.spinner("Training models..."):
                    # Classification
                    if problem_type == "Classification":
                        results_df, trained_models = train_classification_model(ml_data["X_train"], ml_data["X_test"], ml_data["y_train"], ml_data["y_test"],selected_model=(selected_model if training_mode =="Train Selected Model" else None))

                    else:
                        results_df, trained_models = train_regression_model(ml_data["X_train"], ml_data["X_test"], ml_data["y_train"], ml_data["y_test"],selected_model=(selected_model if training_mode =="Train Selected Model" else None))    
                    # Save results
                    st.session_state["model_results"] = results_df
                    st.session_state["trained_models"] = trained_models
                    st.success("Models trained Successfully!")

            except Exception as e:
                st.error(f"Error while training models: {e}")

    # Model Comparison
    if "model_results" in st.session_state:
        st.markdown("## 📊 Model Results")
        results_df = st.session_state["model_results"]
        st.markdown("### 📈 Model Comparison")
        st.caption("Compare the performance of all trained models.")

        st.dataframe(results_df, width="stretch",hide_index=True)

    # Best Model Selection
    if "model_results" in st.session_state:
        results_df = st.session_state["model_results"].copy()
        st.markdown("### Best Model")

        # Classification → highest F1 Score
        if problem_type == "Classification":
            score_column = "F1 Score"

        # Regression → highest R2 Score
        else: 
            score_column = "R2 Score"

        results_df[score_column] = pd.to_numeric(results_df[score_column], errors="coerce")
        valid_results = results_df.dropna(subset= [score_column])

        if valid_results.empty:
            st.error(f"no valid {score_column} values were produced by trained models.")
            st.info("Please check the model traininf results.")
        else:
            best_index = valid_results[score_column].idxmax()
            best_model_name = valid_results.loc[best_index,"Model"]
            best_score = valid_results.loc[best_index,score_column]

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **{best_model_name}**")
            with col2:
                st.metric(score_column, round(float(best_score),4))

            if("trained_models") in st.session_state and best_model_name in st.session_state["trained_models"]:
                st.session_state["best_model_name"] = (best_model_name)
                st.session_state["best_model"] = (st.session_state["trained_models"][best_model_name])
            else:
                st.error("Best model could not be found in the trained models.")

    # Model Evaluation
    if "best_model" in st.session_state:
        best_model = st.session_state["best_model"]
        best_model_name = st.session_state["best_model_name"]

        # Model Persistence
        st.markdown("## 💾 Save Trained Model")
        try:
            model_buffer = BytesIO()
            joblib.dump(best_model,model_buffer)
            model_buffer.seek(0)
            st.download_button(label="Download Trained Model",data= model_buffer,file_name=f"{best_model_name.replace(' ', '_')}.pkl",mime = "application/octet-stream")

        except Exception as e:
            st.error(f"Unable to export trained model: {e}")

        # Model Evaluation
        st.markdown("## 📋 Model Evaluation")
        st.write(f"Evaluation for: **{best_model_name}**")

        # Classification Evaluation
        if problem_type == "Classification":
            evaluation = evaluate_classification_model(best_model, ml_data["X_test"], ml_data["y_test"])


            # Metrics
            st.markdown("### Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            model_results = st.session_state["model_results"]

            best_row = model_results[model_results["Model"] == best_model_name].iloc[0]
            with col1:
                st.metric("Accuracy", best_row["Accuracy"])

            with col2:
                st.metric("Precision", best_row["Precision"])

            with col3:
                st.metric("Recall", best_row["Recall"])

            with col4:
                st.metric("F1 Score", best_row["F1 Score"])

            # Confusion Matrix
            st.markdown("### 🔲 Confusion Matrix")
            cm = evaluation["confusion_matrix"]

            fig, ax = plt.subplots()
            im = ax.imshow(cm)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            ax.set_title("Confusion Matrix")
            ax.set_xticks(range(len(cm)))
            ax.set_yticks(range(len(cm)))

            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(j,i,cm[i,j], ha="center",va="center")

            fig.colorbar(im)
            st.pyplot(fig,width="stretch")
            plt.close(fig)

            # Classification Report 
            st.markdown("### Classification Report")
            report_df = pd.DataFrame(evaluation["classification_report"]).transpose()
            st.dataframe(report_df, width="stretch")

            #ROC-AUC
            roc_auc = evaluation["roc_auc"]
            if roc_auc is not None:
                st.markdown("### ROC-AUC")
                st.metric( "ROC-AUC Score", f"{evaluation['roc_auc']:.3f}")

        #Regression Evaluation
        else:
            evaluation = evaluate_regression_model(best_model, ml_data["X_test"], ml_data["y_test"])
            st.markdown("### 📊 Performance Metrics")

            # Metrics
            col1, col2, col3 = st.columns(3)
                                            
            with col1:
                st.metric("MAE", round(evaluation["mae"],4))
                        
            with col2:
                st.metric("RMSE", round(evaluation["rmse"],4))
                        
            with col3:
                st.metric("R² Score", round(evaluation["r2"],4))

            # Actual vs Predicted
            st.markdown("### Actual vs Predicted")
            actual = np.array(ml_data["y_test"])
            predicted = np.array(evaluation["predictions"])

            fig, ax = plt.subplots()
            ax.scatter(actual,predicted)

            min_value = min(actual.min(),predicted.min())
            max_value = max(actual.max(),predicted.max())

            ax.plot([min_value, max_value],[min_value, max_value],linestyle="--")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")

            st.pyplot(fig,width="stretch")
            plt.close(fig)

            # RESIDUAL PLOT
            st.markdown("### Residual Analysis")
            residuals= np.array(evaluation["residuals"])

            fig, ax = plt.subplots()
            ax.scatter(predicted,residuals)

            ax.axhline(y=0,linestyle="--")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residual Plot")

            st.pyplot(fig,width="stretch")
            plt.close(fig)

    # FEATURE IMPORTANCE
    if "best_model" in st.session_state:
        st.markdown("### Feature importance")
        best_model = st.session_state["best_model"]

        st.caption("Features ranked according to their contribution to the selected model.")

        # get feature names
        feature_names = ml_data["feature_names"]
        importance_df = get_feature_importance(best_model,feature_names)

        if importance_df is not None:
                        
            #Importance Table
            st.dataframe(importance_df, width="stretch", hide_index=True) 
            # Feature Importance Chart
            st.markdown("### Feature Importance Chart")

            chart_df = importance_df.head(10)
            fig, ax = plt.subplots()
            ax.barh(chart_df["Feature"], chart_df["Importance"])

            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")

            ax.set_title("Top 10 Important Features")
            ax.invert_yaxis()
            st.pyplot(fig,width="stretch")
            plt.close(fig)

        else:
            st.info("Feature importance is not available for the selected model.")             

elif page == "About":
    st.markdown("""
    <div class="analysis-section">
    <h1>ℹ️ About DataLens</h1>
    <p class="analysis-description">
        An interactive Data Science platform for exploring, analyzing,
        visualizing and modelling datasets.
    </p>
    </div>
""", unsafe_allow_html=True)
    # What is DataLens
    st.markdown("### 📊 What is DataLens?")
    st.markdown("""
    <div class="info-card">
    <p>
        <b>DataLens</b> is an interactive Data Science platform that allows
        users to upload CSV and Excel datasets and perform data analysis,
        visualization and machine learning tasks in one place.
    </p>
    </div>
""",unsafe_allow_html=True)

    # Features
    st.markdown("### 🚀 What can you do with DataLens?")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🔍 Data Analysis</h3>
                <ul>
                    <li>Data quality analysis</li>
                    <li>Missing value handling</li>
                    <li>Duplicate detection and removal</li>
                    <li>Column type analysis</li>
                    <li>Statistical analysis</li>
                    <li>Categorical analysis</li>
                    <li>Outlier detection</li>
                    <li>Correlation analysis</li>
                </ul>
            </div>""",unsafe_allow_html=True)

    with col2:
        st.markdown("""
         <div class="info-card">
                <h3>🤖 Data Science & ML</h3>
                <ul>
                    <li>Interactive visualizations</li>
                    <li>Classification models</li>
                    <li>Regression models</li>
                    <li>Model comparison</li>
                    <li>Model evaluation</li>
                    <li>Feature importance</li>
                    <li>Trained model export</li>
                    <li>Processed data export</li>
                </ul>
            </div>""",unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")

    st.markdown("""
        <div class="tech-stack">
            <p>
                <b>Programming:</b> Python
            </p>
            <p>
                <b>Data Processing:</b> Pandas • NumPy
            </p>
            <p>
                <b>Machine Learning:</b> Scikit-learn
            </p>
            <p>
                <b>Visualization:</b> Matplotlib
            </p>
            <p>
                <b>Application:</b> Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "💡 DataLens provides an end-to-end workflow: "
        "Upload → Explore → Clean → Visualize → Model → Evaluate → Export"
    )

elif page == "Instructions":

    st.markdown("""
        <div class="analysis-section">
            <h1>📖 How to Use DataLens</h1>
            <p class="analysis-description">
                Follow the workflow below to explore, analyze and model your dataset.
            </p>
        </div>""",unsafe_allow_html=True)
    # Step1

    st.markdown("""
         <div class="instruction-step">
            <div class="step-number">1️⃣ Upload Your Dataset</div>
            <p>
                Use the <b>Dataset</b> section in the sidebar to upload a
                <b>CSV</b> or <b>Excel (.xlsx)</b> file.
                DataLens validates the dataset before processing it.
            </p>
        </div>""",unsafe_allow_html=True)

    # Step2
    st.markdown("""
        <div class="instruction-step">
            <div class="step-number">2️⃣ Explore the Overview</div>
            <p>
                Open the <b>Overview</b> page to view rows, columns,
                missing values, duplicate rows, dataset preview and
                dataset information.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # step3
    st.markdown("""
        <div class="instruction-step">
            <div class="step-number">3️⃣ Clean and Analyze Your Data</div>
            <p>
                Use the <b>Analysis</b> page to examine and improve
                the quality of your dataset.
            </p>
            <ul>
                <li>Missing value analysis and handling</li>
                <li>Duplicate detection and removal</li>
                <li>Column type analysis</li>
                <li>Data type conversion</li>
                <li>Statistical analysis</li>
                <li>Categorical analysis</li>
                <li>Outlier analysis</li>
                <li>Correlation analysis</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # step4
    st.markdown("""
        <div class="instruction-step">
            <div class="step-number">4️⃣ Visualize Your Data</div>
            <p>
                Use the <b>Visualizations</b> page to understand
                distributions, relationships and patterns.
            </p>
            <ul>
                <li>Histogram</li>
                <li>Bar Chart</li>
                <li>Box Plot</li>
                <li>Scatter Plot</li>
                <li>Correlation Heatmap</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # step5
    st.markdown("""
        <div class="instruction-step">
            <div class="step-number">5️⃣ Build Machine Learning Models</div>
            <p>
                Open <b>ML Models</b> after preparing your dataset.
                Select the target column, features and problem type.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Models
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
     <div class="info-card">
                <h3>🎯 Classification</h3>
                <ul>
                    <li>Logistic Regression</li>
                    <li>Random Forest</li>
                    <li>Gradient Boosting</li>
                </ul>
            </div>""",unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div class="info-card">
                <h3>📈 Regression</h3>
                <ul>
                    <li>Linear Regression</li>
                    <li>Random Forest</li>
                    <li>Gradient Boosting</li>
                </ul>
            </div>""",unsafe_allow_html=True)

    # step6
    st.markdown("""
     <div class="instruction-step">
            <div class="step-number">6️⃣ Evaluate and Export</div>
            <p>
                DataLens evaluates the selected model using appropriate
                performance metrics and allows you to download the
                trained model and processed dataset.
            </p>
        </div>""",unsafe_allow_html=True)

    st.success("💡 Recommended workflow: Upload → Overview → Analysis → "
                "Visualizations → ML Models")

elif page == "Help":
    st.markdown("""
        <div class="analysis-section">
            <h1>🆘 Help & Frequently Asked Questions</h1>
            <p class="analysis-description">
                Find answers to common questions about using DataLens.
            </p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset","🔍 Analysis","📈 Visualization","🤖 Machine Learning"])
    with tab1:
        st.markdown("### What files can I upload?")
        st.write("DataLens currently supports CSV and Excel (.xlsx) files.")
        st.markdown("### Why can't I upload my file?")
        st.write("""Make sure the file is a supported format and contains a valid tabular dataset with rows and columns.""")

    with tab2:
        st.markdown("### What happens when I clean my data?")
        st.write(""" DataLens keeps the original uploaded dataset unchanged and performs cleaning operations on a separate processed dataset.""")

        st.markdown("### Can I remove missing values?")
        st.write("""Yes. You can remove rows containing missing values or use appropriate imputation methods depending on the column type.""")

        st.markdown("### What is an outlier?")
        st.write("""An outlier is a value that is unusually far from the typical distribution of a numerical variable. DataLens uses the IQR method for outlier detection.""")

    with tab3:
        st.markdown("### Why don't I see my cleaned data?")
        st.write("""Visualizations use the processed dataset. Therefore, changes made in the Analysis page should be reflected in subsequent visualizations.""")

        st.markdown("### Why can't I select a column?")
        st.write("""Some visualizations require specific column types. For example, histograms and scatter plots require numerical columns.""")

    with tab4:
        st.markdown("### Which problem types are supported?")
        st.write("""DataLens currently supports:  Classification Regression""")

        st.markdown("### Which models are available?") 
        st.write("""
                      **Classification:** Logistic Regression, Random Forest, Gradient Boosting.

                        **Regression:** Linear Regression, Random Forest, Gradient Boosting.""")

        st.markdown("### Why did model training fail?")
        st.write("""Model training can fail when the selected target or features are unsuitable, when there are insufficient samples, or when the target contains invalid or problematic values.""")

elif page == "Contact":
    st.markdown("""
        <div class="analysis-section">
            <h1>📩 Contact</h1>
            <p class="analysis-description">
                Have questions, suggestions or feedback about DataLens?
                Let's connect.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👋 Let's Connect")
    st.write("""If you have questions, suggestions, or would like to discuss DataLens, you can reach out through the links below.""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🐙 GitHub**")
        st.link_button("View DataLens Repository","https://github.com/anshika092004/Datalens")

    with col2:
        st.markdown("**💼 LinkedIn**")
        st.link_button("Connect on LinkedIn","https://linkedin.com/in/anshika092004")

    st.markdown("---")
    st.info("Thank you for using DataLens!")




                