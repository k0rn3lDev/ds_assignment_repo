from sklearn.neighbors import NearestNeighbors

# Step 1: Create the dataset
# Each person has 2 features: Age and Years of Experience
age_experience_data = [
    [22, 2],  # Person 1
    [30, 5],  # Person 2
    [25, 3],  # Person 3
    [28, 4],  # Person 4
    [35, 10], # Person 5
    [24, 1],  # Person 6
    [32, 6]   # Person 7
]

# Salaries corresponding to each person
salaries = [
    30000,  # Person 1
    45000,  # Person 2
    38000,  # Person 3
    42000,  # Person 4
    70000,  # Person 5
    28000,  # Person 6
    50000   # Person 7
]

# Step 2: Define the new person
new_person = [[26, 3]]  # Age 26, 3 years of experience

# Step 3: Create and fit the KNN model
# We ask for n_neighbors = 7 (because part B asks for 7 neighbors)
knn = NearestNeighbors(n_neighbors=7, metric='euclidean')  # Use Euclidean distance
knn.fit(age_experience_data)  # Fit the model with the data

# Step 4: Find the 7 nearest neighbors
distances, indices = knn.kneighbors(new_person)

# Step 5: Print distances, labels (salaries), and indices
print("Distances to neighbors:", distances[0])
print("Indices of neighbors:", indices[0])
print("Salaries (labels) of neighbors:", [salaries[i] for i in indices[0]])

# Step 6: (Optional) Predict salary based on the 5 nearest neighbors for part A
# Find 5 nearest neighbors
knn_5 = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_5.fit(age_experience_data)
distances_5, indices_5 = knn_5.kneighbors(new_person)

# Get their salaries
salaries_5 = [salaries[i] for i in indices_5[0]]

# Predict the salary by taking the mean of the 5 nearest salaries
predicted_salary = sum(salaries_5) / len(salaries_5)
print("Predicted Salary for new person (based on 5 neighbors):", predicted_salary)