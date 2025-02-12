import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("netflix_titles.csv") 
print(df.head())
print(df.isnull().sum())
print(df.info())       
print(df.describe())    

plt.figure(figsize=(6, 6))
sns.countplot(data=df, x="type", color="lightblue")
plt.title("Distribution of Movies and TV Shows")
plt.show()

top_countries = df["country"].value_counts().head(10)
plt.figure(figsize=(6,6))
sns.barplot(x=top_countries.values, y=top_countries.index, color="lightgreen")
plt.title("Top 10 Countries with Most Netflix Titles")
plt.xlabel("Number of Titles")
plt.show()

df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
df["year_added"] = df["date_added"].dt.year.fillna(df["release_year"])

year_counts = df["year_added"].value_counts().sort_index()
plt.figure(figsize=(10, 4))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o")
plt.title("Number of Titles Added to Netflix Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Titles")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode "type" (Movie = 0, TV Show = 1)
label_encoder = LabelEncoder()
df["type_encoded"] = label_encoder.fit_transform(df["type"])

# Extract relevant numerical features
df["year_added"] = df["year_added"].astype(int)
df["release_year"] = df["release_year"].astype(int)

# One-hot encoding categorical features (rating, country, genres)
df_encoded = pd.get_dummies(df, columns=["rating", "country", "listed_in"], drop_first=True)

# Select features and target
features = ["release_year", "year_added"] + list(df_encoded.columns[13:])
X = df_encoded[features]
y = df["type_encoded"]

from sklearn.ensemble import RandomForestClassifier

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

