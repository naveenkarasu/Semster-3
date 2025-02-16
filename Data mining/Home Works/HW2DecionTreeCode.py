import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Transcribing the data from the image into a DataFrame
data = pd.DataFrame({
    'Department': ['Sales', 'Sales', 'Sales', 'Systems', 'Systems', 'Systems', 'Marketing', 'Marketing', 'Secretary', 'Secretary'],
    'Status': ['senior', 'junior', 'junior', 'junior', 'senior', 'junior', 'senior', 'junior', 'senior', 'junior'],
    'Age_Range': ['31..35', '26..30', '31..35', '21..25', '31..35', '26..30', '36..40', '31..35', '46..50', '26..30'],
    'Salary_Range': ['46K..50K', '26K..30K', '31K..35K', '46K..50K', '66K..70K', '46K..50K', '46K..50K', '41K..45K', '36K..40K', '26K..30K'],
    'Count': [30, 40, 40, 20, 5, 3, 10, 4, 4, 6]
})

# Preprocess the data
data['Department'] = data['Department'].astype('category').cat.codes
data['Status'] = data['Status'].astype('category').cat.codes

# Convert salary range to average salary for simplicity
def parse_salary_range(salary_range):
    low, high = salary_range.replace('K', '').split('..')
    return (int(low) + int(high)) / 2 * 1000

data['Salary'] = data['Salary_Range'].apply(parse_salary_range)
data.drop(['Age_Range', 'Salary_Range'], axis=1, inplace=True)

# Define feature matrix X and target vector y
X = data[['Department', 'Status', 'Salary']]
y = data['Count']

# Initialize and train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X, y)

# Visualize the decision tree with larger plot size and text size
plt.figure(figsize=(20, 10))  # Adjust the size of the figure
tree.plot_tree(decision_tree, filled=True, feature_names=['Department', 'Status', 'Salary'], class_names=True,
               fontsize=12)  # Increase the fontsize for readability
plt.show()
