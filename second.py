
# Importing essential libraries for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt

# Load the Titanic dataset into a DataFrame
titanic_data = pd.read_csv('train.csv')

# Check for missing values in the dataset
missing_values = titanic_data.isnull().sum()

# Fill missing age values with the median age
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)

# Fill missing embarked values with the mode (most frequent value)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Optionally replace missing cabin values with 'Unknown'
titanic_data['Cabin'].fillna('Unknown', inplace=True)

# Create a new column for age categories
titanic_data['Age_Category'] = pd.cut(titanic_data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'Adult', 'Middle Aged', 'Senior'])

# Filter for female passengers who survived
female_survivors = titanic_data[(titanic_data['Sex'] == 'female') & (titanic_data['Survived'] == 1)]

# Filter for passengers older than 50
older_passengers = titanic_data[titanic_data['Age'] > 50]

# Filter for children (younger than 15) who survived
children_survivors = titanic_data[(titanic_data['Age'] < 15) & (titanic_data['Survived'] == 1)]

# Plotting a bar chart for survival rates by passenger class
survival_rate_by_class = titanic_data.groupby('Pclass')['Survived'].mean()

plt.figure(figsize=(8, 5))
survival_rate_by_class.plot(kind='bar', color='skyblue')
plt.title('Survival Rate by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

# Histogram for age distribution of all passengers
plt.figure(figsize=(10, 5))
plt.hist(titanic_data['Age'].dropna(), bins=30, color='lightgreen', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()