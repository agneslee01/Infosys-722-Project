
"""

Iteration 3
Name : Agnes Lee (246046668)

"""

"""

Step 1 : Business Understanding

"""


"""

Step 2 : Data Understanding

"""

# Load the csv file into Spyder

file = 'heart_attack_dataset.csv'

import pandas as pd
heart_data = pd.read_csv(file)


# Basic information about the dataset 
heart_data.info()


# shows the count, mean, max...
heart_summary_stat = heart_data.describe().transpose()    


# summary statistics for numerical variables only + transpose
import numpy as np

summary_stats = round(heart_data.select_dtypes(include=[np.int64, np.float64]).describe().transpose(),2)

binary_var = ['Diabetes', 'Family History (1: Yes)', 'Smoking','Obesity', 'Alcohol Consumption',\
              'Previous Heart Problems (1 : Yes)', 'Medication Use','Heart Attack Risk (1: Yes)']

filtered_summary_stats = summary_stats.drop(binary_var)
                      

# Value counts
heart_data['Sex'].value_counts()


# Group_by
heart_data[['Income','Age','Sex']].groupby('Sex').agg('mean')


# Explore the data
import matplotlib.pyplot as plt
heart_data['Country'].value_counts()[heart_data['Country'].unique()].plot.bar()


# Get % of gender
gender_count = heart_data['Sex'].value_counts()
gender_percentages = (gender_count / gender_count.sum()) * 100

gender_percentages.plot(kind='bar', ylabel = "Percentage", title = 'Percentage Distribution of Gender')


# Bar grah for stress level for each gender 

count_data = heart_data.groupby(['Stress Level', 'Sex']).size().unstack(fill_value=0)

count_data.plot(kind='bar', figsize=(10, 6) ,ylabel = "Count", title = 'Count of Stress Levels by Gender')


# Pie chart for diet variable
diet_counts = heart_data['Diet'].value_counts()

plt.pie(diet_counts, labels=diet_counts.index, autopct='%1.1f%%', startangle=140)


# Scatter plot for Age vs Income

#heart_data.sample(n=100).plot.scatter(x = 'Age', y = 'Income')


# Check for missing values
missing_values = heart_data.isna().sum()
print(missing_values[missing_values > 0])


# Outlier detection

numeric_columns = heart_data.select_dtypes(include=['int', 'float']).columns

Q1 = heart_data[numeric_columns].quantile(0.25)
Q3 = heart_data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
outliers = ((heart_data[numeric_columns] < (Q1 - threshold * IQR)) | (heart_data[numeric_columns] > (Q3 + threshold * IQR)))

print(outliers.any())
outlier_counts = outliers.sum()
print(outlier_counts)
outlier_counts[outlier_counts > 0]

# Check the smoking variable count
heart_data['Smoking'].value_counts()


# Extreme values
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR
IQR = Q3 - Q1
extreme_value = heart_data[(heart_data[numeric_columns] < lower_bound) | (heart_data[numeric_columns] > upper_bound)]

extreme_counts = extreme_value.sum()
print(extreme_counts)
extreme_counts[extreme_counts > 0]

###############################################################################################

""""

Step 3 : Data Preparation

"""

# Drop PatientID column

heart_data = heart_data.drop(['Patient ID'], axis = 1)
heart_data.info()


# Target count
heart_data['Heart Attack Risk (1: Yes)'].value_counts()


# Impute missing value (mode)

# Calculate the mode of the 'stress level' variable
stress_mode = heart_data['Stress Level'].mode()[0]

heart_data['Stress Level'] = heart_data['Stress Level'].fillna(stress_mode)

missing_values_after_imputation = heart_data['Stress Level'].isnull().sum()

print(missing_values_after_imputation)


# Impute missing value (mean)
exercise_mean = heart_data['Exercise Hours Per Week'].mean()
sedentary_mean = heart_data['Sedentary Hours Per Day'].mean()
bmi_mean = heart_data['BMI'].mean()
sleep_mean = heart_data['Sleep Hours Per Day'].mean()

heart_data['Exercise Hours Per Week'] = heart_data['Exercise Hours Per Week'].fillna(exercise_mean)
heart_data['Sedentary Hours Per Day'] = heart_data['Sedentary Hours Per Day'].fillna(sedentary_mean)
heart_data['BMI'] = heart_data['BMI'].fillna(bmi_mean)
heart_data['Sleep Hours Per Day'] = heart_data['Sleep Hours Per Day'].fillna(sleep_mean)


# CHeck for missing values
missing_values_1 = heart_data.isna().sum()
print(missing_values_1[missing_values_1 > 0])
heart_data.info()


# Remove extreme values
cleaned_data = heart_data.drop(heart_data[heart_data['Stress Level'] == 20.0].index)
cleaned_data = cleaned_data.drop(cleaned_data[cleaned_data['Sleep Hours Per Day'] == 20.0].index)
cleaned_data = cleaned_data.drop(cleaned_data[cleaned_data['Heart Rate'] == 200.0].index)
cleaned_data = cleaned_data.drop(cleaned_data[cleaned_data['Exercise Hours Per Week'] == 40.546388].index)

cleaned_data.info()

# cheking for extreme values after removing them
extreme_value_1 = cleaned_data[(cleaned_data[numeric_columns] < lower_bound) | (cleaned_data[numeric_columns] > upper_bound)]

extreme_counts1 = extreme_value_1.sum()
print(extreme_counts1)
extreme_counts1[extreme_counts1 > 0]


# Create new variable - Age_group

# Define age categories and corresponding labels
bins = [18, 30, 60, 103]  
labels = ['Young Adults', 'Middle Age', 'Old']

cleaned_data['Age group'] = pd.cut(cleaned_data['Age'], bins=bins, labels=labels, right=False)

cleaned_data.info()

cleaned_data['Age group'].value_counts()

import seaborn as sns
sns.countplot(x = "Age group", data = cleaned_data)
plt.title('Distribution of Age Group')
plt.show()


# Balance out the weight for target
target_count = cleaned_data['Heart Attack Risk (1: Yes)'].value_counts()
target_percentages = (target_count / target_count.sum()) * 100
print(target_percentages)


import pandas as pd
import numpy as np



# Separate majority and minority classes
majority_class = target_count.idxmax()
minority_class = target_count.idxmin()

# Filter majority and minority classes
majority_data = cleaned_data[cleaned_data['Heart Attack Risk (1: Yes)'] == majority_class]
minority_data = cleaned_data[cleaned_data['Heart Attack Risk (1: Yes)'] == minority_class]

# Calculate the difference in counts between the two classes
count_difference = target_count[majority_class] - target_count[minority_class]

# Randomly sample instances from the majority class to match the minority class
balanced_majority_data = majority_data.sample(target_count[minority_class], replace=False, random_state=42)

# Concatenate the balanced majority data with the minority data
balanced_data = pd.concat([balanced_majority_data, minority_data])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the balanced dataset
print(balanced_data.head())

# Check the class distribution of the balanced dataset
print(balanced_data['Heart Attack Risk (1: Yes)'].value_counts())

# Save the balanced dataset to a new CSV file
balanced_data.to_csv('balanced_heart_data.csv', index=False)


# Load the new csv file into Spyder
file1 = 'balanced_heart_data.csv'

import pandas as pd
new_heart_data = pd.read_csv(file1)
new_heart_data.info()


# Merge of 2 dataset.
file2 = 'heart_data_10%(1).csv'
file3 = 'heart_data_10%(2).csv'

import pandas as pd
merge_1 = pd.read_csv(file2)
merge_2 = pd.read_csv(file3)

print(merge_1.shape)
print(merge_2.shape)

# Concatenate  along axis=0 (rows)
merged_data = pd.concat([merge_1, merge_2], axis=0, ignore_index=True)

print(merged_data.shape)

merged_data.info()

# Save the merged dataset to a new CSV file
merged_data.to_csv('merged_data.csv', index=False)

merged_data_copy = merged_data.copy()

###############################################################################################


""""

Step 4 : Data Transformation

"""


# Perform encoding for 'Gender' column
merged_data['Sex'] = merged_data['Sex'].replace({'Male': 0, 'Female': 1})

from sklearn.preprocessing import LabelEncoder
categorical_columns = merged_data.select_dtypes(include=['object']).columns.tolist()

# Dictionary to store LabelEncoder instances
label_encoders = {}

# Apply label encoding to each categorical column
for col in categorical_columns:
    # Initialize LabelEncoder for the current column
    label_encoder = LabelEncoder()
    merged_data[col + '_Encoded'] = label_encoder.fit_transform(merged_data[col])
    label_encoders[col] = label_encoder

# Display the DataFrame after label encoding
print("Encoded DataFrame:")
print(merged_data)

# Inspect the mapping of encoded values to original categories for each column
for col in categorical_columns:
    print(f"\nMapping for '{col}':")
    label_mapping = {encoded_value: original_category for original_category, encoded_value in zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_))}
    print(label_mapping)
    
# Drop original categorical columns after label encoding
merged_data.drop(merged_data.select_dtypes(include=['object']).columns, axis=1, inplace=True)



# Feature selection     
    
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression
import numpy as np

# Split into features (X) and target variable (y) if applicable
X = merged_data.drop('Heart Attack Risk (1: Yes)', axis=1)  
y = merged_data['Heart Attack Risk (1: Yes)']


# Assuming X contains your features and y is your target variable
selector = SelectKBest(f_regression, k=12)
X_selected = selector.fit(X,y)
# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)
# Get the names of the selected features
selected_feature_names = X.columns[selected_feature_indices].tolist()
# Get the scores of all features based on f_regression
scores = selector.scores_
# Create a DataFrame to display feature names and their scores
feature_scores_df = pd.DataFrame({'Feature': X.columns.tolist(), 'Score': scores})
feature_scores_df.sort_values(by='Score', ascending=False, inplace=True)

# Filter the DataFrame to display only selected features and their scores
selected_features_with_scores_df = feature_scores_df[feature_scores_df['Feature'].isin(selected_feature_names)]
print(selected_features_with_scores_df)


# Filter out the features we do not want
# List of columns to keep
columns_to_keep = ['Heart Attack Risk (1: Yes)', 'Diabetes', 'Systolic','Age group_Encoded',\
                   'Cholesterol', 'Stress Level','Previous Heart Problems (1 : Yes)','Smoking',\
                    'Income']

new_data = merged_data[columns_to_keep]
new_data.info()

# Save this new df 
new_data.to_csv('new_data.csv', index=False)



# Distribution of target variable
new_data['Heart Attack Risk (1: Yes)'].value_counts()


# Read in the new data (less variables)
file4 = 'new_data.csv'
new_data = pd.read_csv(file4)


# Check the std dev for numerical variables
std_dev = np.std(new_data)
print("Standard Deviation (NumPy):", std_dev)

# Log transformatio of Income (example)
# Compute logarithm of income using numpy.log()
new_data['Log_Income'] = np.log(new_data['Income'])

# Calculate standard deviation of log-transformed income
log_income_std_dev = new_data['Log_Income'].std()
print("Standard Deviation of Log-transformed Income:", log_income_std_dev)

# Drop the log(income) column
new_data = new_data.drop('Log_Income', axis=1)


###############################################################################################


"""

Step 6 : Algorithm Selection

"""

# Feature selection - check no. of fields
new_data.shape


x1 = new_data.drop('Heart Attack Risk (1: Yes)', axis=1)  
y1 = new_data['Heart Attack Risk (1: Yes)']




# Decision Tree (CART)
#from sklearn.tree import DecisionTreeClassifier, export_text
# Instantiate and train Decision Tree classifier (CART) using all data
#tree_classifier = DecisionTreeClassifier()
#tree_classifier.fit(x1, y1)
#tree_rules = export_text(tree_classifier, feature_names=list(x1.columns))
#print("Decision Tree Rules:\n", tree_rules)




# make a copy so that apriori can work on this data instead
apriori_data_copy = new_data.copy()




"""

Step 6 & 7 : Data Mining

"""
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x1,y1,test_size=0.2, random_state=42)


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Instantiate Logistic Regression model
logreg = LogisticRegression(random_state=42)
# fit the model on the training data
logreg.fit(X_train, Y_train)
# Make predictions on the testing data
y_pred = logreg.predict(X_test)
# Make predictions on the training data
y_pred_train = logreg.predict(X_train)
# Evaluate model performance
accuracy = accuracy_score(Y_test, y_pred)
accuracy_train = accuracy_score(Y_train, y_pred_train)
print(f"Accuracy on test set: {accuracy:.2f}")
print(f"Accuracy on train set: {accuracy_train:.2f}")

# Retrieve coefficient magnitudes and corresponding feature names
coef_magnitudes = logreg.coef_[0]
feature_names = x1.columns
# Create a DataFrame to display coefficients and feature names
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef_magnitudes})
# Sort coefficients by magnitude (absolute value) in descending order
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
# Display the coefficients and feature names
print("Coefficient Magnitudes:")
print(coef_df)

# Create a cross-tabulation of predicted vs. actual classes
crosstab = pd.crosstab(index=Y_test, columns=y_pred, rownames=['Actual'], colnames=['Predicted'])
print(crosstab)

########################################################################################

# Decision Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
dt_classifier = DecisionTreeClassifier(random_state=42)
# Fit the classifier on the training data
dt_classifier.fit(X_train, Y_train)
# Predict the labels for the test set
y_pred_dt = dt_classifier.predict(X_test)
# Predict the labels for the train set
y_pred_train_dt = dt_classifier.predict(X_train)
# Calculate the accuracy score
accuracy1 = accuracy_score(Y_test, y_pred_dt)
accuracy_train1 = accuracy_score(Y_train, y_pred_train_dt)
print(f"Accuracy Score for test set: {accuracy1:.4f}")
print(f"Accuracy Score for train set: {accuracy_train1:.4f}")

# Retrieve feature importances
feature_importances = dt_classifier.feature_importances_
# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': x1.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# Display feature importances
print("Feature Importances:")
print(importance_df)


# Generate text representation of the decision tree rules
tree_rules_text = export_text(dt_classifier, feature_names=list(x1.columns))
print("Decision Tree Rules:")
print(tree_rules_text)

# Create a cross-tabulation of predicted vs. actual classes
crosstab_dt = pd.crosstab(index=Y_test, columns=y_pred_dt, rownames=['Actual'], colnames=['Predicted'])
# Print the cross-tabulation
print("Cross-Tabulation (Actual vs. Predicted):\n")
print(crosstab_dt)



########################################################################################

#Apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# Convert variables to binary
# make a copy so that apriori can work on this data instead
apriori_data_copy = new_data.copy()
#apriori_data_copy['Stress Level'] = apriori_data_copy['Stress Level'].apply(lambda x: 0 if x <= 5 else 1)
apriori_data_copy['Cholesterol'] = apriori_data_copy['Cholesterol'].apply(lambda x: 0 if x <= 240 else 1)
apriori_data_copy['Systolic'] = apriori_data_copy['Systolic'].apply(lambda x: 0 if x <= 120 else 1)
apriori_data_copy['Income'] = apriori_data_copy['Income'].apply(lambda x: 0 if x <= 100000 else 1)
apriori_data_copy = apriori_data_copy.drop('Age group_Encoded', axis=1)
apriori_data_copy = apriori_data_copy.drop('Stress Level', axis=1)


# Find frequent itemsets with minimum support threshold
frequent_itemsets = apriori(apriori_data_copy, min_support=0.2, use_colnames=True)
# Display frequent itemsets
#print(frequent_itemsets)
# Generate association rules from frequent itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
specific_rules = rules[rules['consequents'] == {'Heart Attack Risk (1: Yes)'}]

# Display association rules
print(specific_rules)

########################################################################################


# 2-step clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score

# Define the number of clusters (K)
num_clusters = 3


# Create an instance of AgglomerativeClustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    

# Fit the clustering model to the train data
train_cluster_labels = agglomerative_clustering.fit_predict(X_train)

# Add cluster labels to the original training dataset
train_data_with_clusters = pd.concat([X_train, pd.Series(train_cluster_labels, name='Cluster')], axis=1)

# Reshape the training dataset for seaborn boxplot
train_data_melted = train_data_with_clusters.melt(id_vars='Cluster', var_name='Feature', value_name='Value')


# Iterate over each feature and create separate boxplots for each
for feature in X_train.columns:
    # Reshape the training dataset for seaborn boxplot
    train_data_melted = train_data_with_clusters.melt(id_vars='Cluster', value_vars=feature, var_name='Feature', value_name='Value')
    
    # Plot boxplot for the current feature across clusters
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Cluster', y='Value', data=train_data_melted)
    plt.title(f'Feature: {feature} - Distribution Across Clusters (Training Set)')
    plt.xlabel('Cluster')
    plt.ylabel('Feature Value')
    plt.show()



# Assign clusters to the test data
test_cluster_labels = agglomerative_clustering.fit_predict(X_test)  # Using fit_predict on test data to assign clusters

# Compute silhouette score for training data
train_silhouette_score = silhouette_score(X_train, train_cluster_labels)
print(f"Silhouette Score (Training): {train_silhouette_score}")

# Compute silhouette score for test data
test_silhouette_score = silhouette_score(X_test, test_cluster_labels)
print(f"Silhouette Score (Test): {test_silhouette_score}")


# Count the frequency of each cluster label
cluster_counts = np.bincount(train_cluster_labels)


# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=range(num_clusters), autopct='%1.1f%%', startangle=140)
plt.title('Cluster Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

##############################################################################################################
# Display cluster counts
print("Cluster Counts:")
for cluster_id, count in enumerate(cluster_counts):
    print(f"Cluster {cluster_id}: {count} samples")

##############################################################################################################


"""

Step 8 : Iteration

"""

# 1st iteration
columns_to_drop = ['Heart Attack Risk (1: Yes)', 'Age group_Encoded', 'Diabetes']

x2 = new_data.drop(columns=columns_to_drop, axis=1)  
y2 = new_data['Heart Attack Risk (1: Yes)']

X_train_1,X_test_1,Y_train_1,Y_test_1 = train_test_split(x2,y2,test_size=0.2, random_state=42)

# Instantiate Logistic Regression model
logreg1 = LogisticRegression(random_state=42)
# fit the model on the training data
logreg1.fit(X_train_1, Y_train_1)
# Make predictions on the testing data
y_pred1 = logreg1.predict(X_test_1)
# Make predictions on the training data
y_pred_train1 = logreg1.predict(X_train_1)
# Evaluate model performance
accuracy_lg = accuracy_score(Y_test_1, y_pred1)
accuracy_train_lg = accuracy_score(Y_train_1, y_pred_train1)
print(f"Accuracy on test set: {accuracy_lg:.2f}")
print(f"Accuracy on train set: {accuracy_train_lg:.2f}")

# Retrieve coefficient magnitudes and corresponding feature names
coef_magnitudes_1 = logreg1.coef_[0]
feature_names1 = x2.columns
# Create a DataFrame to display coefficients and feature names
coef_df1 = pd.DataFrame({'Feature': feature_names1, 'Coefficient': coef_magnitudes_1})
# Sort coefficients by magnitude (absolute value) in descending order
coef_df1 = coef_df1.sort_values(by='Coefficient', ascending=False)
# Display the coefficients and feature names
print("Coefficient Magnitudes:")
print(coef_df1)

# Create a cross-tabulation of predicted vs. actual classes
crosstab1 = pd.crosstab(index=Y_test_1, columns=y_pred1, rownames=['Actual'], colnames=['Predicted'])
print(crosstab1)


################################################################################################

# 2nd iteration - Boost

# new_data is  DataFrame
boost_factor = 1.0  # Boost factor of 100% (1.0 for doubling the dataset)
num_boosted_samples = int(len(new_data) * boost_factor)

# Randomly sample existing rows with replacement
boosted_samples = new_data.sample(n=num_boosted_samples, replace=True)

# Update the 'Dementia' and 'Cognitive_Test_Scores' columns with random values
boosted_samples['Heart Attack Risk (1: Yes)'] = np.random.choice([0, 1], size=num_boosted_samples)


# Concatenate the boosted samples with the original dataset
new_data_boosted = pd.concat([new_data, boosted_samples], ignore_index=True)

# Check the shape before and after boosting
print("Original shape:", new_data.shape)
print("Boosted shape:", new_data_boosted.shape)


x3 = new_data_boosted.drop(columns=columns_to_drop, axis=1)  
y3 = new_data_boosted['Heart Attack Risk (1: Yes)']

X_train_2,X_test_2,Y_train_2,Y_test_2 = train_test_split(x3,y3,test_size=0.2, random_state=42)

# Instantiate Logistic Regression model
logreg2 = LogisticRegression(random_state=42)
# fit the model on the training data
logreg2.fit(X_train_2, Y_train_2)
# Make predictions on the testing data
y_pred2 = logreg1.predict(X_test_2)
# Make predictions on the training data
y_pred_train2 = logreg2.predict(X_train_2)
# Evaluate model performance
accuracy_lg1 = accuracy_score(Y_test_2, y_pred2)
accuracy_train_lg1 = accuracy_score(Y_train_2, y_pred_train2)
print(f"Accuracy on test set: {accuracy_lg1:.2f}")
print(f"Accuracy on train set: {accuracy_train_lg1:.2f}")

# Retrieve coefficient magnitudes and corresponding feature names
coef_magnitudes_2 = logreg2.coef_[0]
feature_names2 = x3.columns
# Create a DataFrame to display coefficients and feature names
coef_df2 = pd.DataFrame({'Feature': feature_names2, 'Coefficient': coef_magnitudes_2})
# Sort coefficients by magnitude (absolute value) in descending order
coef_df2 = coef_df2.sort_values(by='Coefficient', ascending=False)
# Display the coefficients and feature names
print("Coefficient Magnitudes:")
print(coef_df2)

# Create a cross-tabulation of predicted vs. actual classes
crosstab2 = pd.crosstab(index=Y_test_2, columns=y_pred2, rownames=['Actual'], colnames=['Predicted'])
print(crosstab2)

################################################################################################

#3rd iteration 

x4 = new_data_boosted.drop('Heart Attack Risk (1: Yes)', axis=1)  
y4 = new_data_boosted['Heart Attack Risk (1: Yes)']

X_train_3,X_test_3,Y_train_3,Y_test_3 = train_test_split(x4,y4,test_size=0.2, random_state=42)

# Instantiate Logistic Regression model
logreg3 = LogisticRegression(random_state=42)
# fit the model on the training data
logreg3.fit(X_train_3, Y_train_3)
# Make predictions on the testing data
y_pred3 = logreg3.predict(X_test_3)
# Make predictions on the training data
y_pred_train3 = logreg3.predict(X_train_3)
# Evaluate model performance
accuracy_lg2 = accuracy_score(Y_test_3, y_pred3)
accuracy_train_lg2 = accuracy_score(Y_train_3, y_pred_train3)
print(f"Accuracy on test set: {accuracy_lg2:.2f}")
print(f"Accuracy on train set: {accuracy_train_lg2:.2f}")

# Retrieve coefficient magnitudes and corresponding feature names
coef_magnitudes_3 = logreg3.coef_[0]
feature_names3 = x4.columns
# Create a DataFrame to display coefficients and feature names
coef_df3 = pd.DataFrame({'Feature': feature_names3, 'Coefficient': coef_magnitudes_3})
# Sort coefficients by magnitude (absolute value) in descending order
coef_df3 = coef_df3.sort_values(by='Coefficient', ascending=False)
# Display the coefficients and feature names
print("Coefficient Magnitudes:")
print(coef_df3)

# Create a cross-tabulation of predicted vs. actual classes
crosstab3 = pd.crosstab(index=Y_test_3, columns=y_pred3, rownames=['Actual'], colnames=['Predicted'])
print(crosstab3)