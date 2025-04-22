import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Preprocess Dataset #
df = pd.read_csv('anime-dataset-2023.csv')

# Handle missing values and unknowns
df = df[df['Score'].notnull()]
df = df[df['Score'] != "UNKNOWN"]
df['Score'] = df['Score'].astype(float)

df['Scored By'] = pd.to_numeric(df['Scored By'], errors='coerce')
df = df[df['Scored By'] >= 100]

# Airing Year extraction
df['Aired'] = df['Aired'].astype(str)
df['Airing_Year'] = df['Aired'].str.extract(r'(\d{4})')
df['Airing_Year'] = pd.to_numeric(df['Airing_Year'], errors='coerce')
df = df[df['Airing_Year'].notnull()]

# Episodes cleanup
df['Episodes'] = df['Episodes'].replace('UNKNOWN', np.nan)
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')
df['Episodes'] = df['Episodes'].fillna(df['Episodes'].median())

# One-hot encode Type and Source
df = pd.get_dummies(df, columns=['Type', 'Source'])

# Studios - keep top 20
top_studios = df['Studios'].value_counts().nlargest(20).index
df['Studios'] = df['Studios'].apply(lambda x: x if x in top_studios else 'Other')
df = pd.get_dummies(df, columns=['Studios'])

# Genres - create binary flags
df['Genres'] = df['Genres'].fillna("")
unique_genres = set(g.strip() for sublist in df['Genres'].str.split(',') for g in sublist)
for genre in unique_genres:
    df[f'Genre_{genre}'] = df['Genres'].str.contains(genre).astype(int)

# Final feature selection
features = ['Episodes', 'Airing_Year'] + \
           [col for col in df.columns if col.startswith('Type_')] + \
           [col for col in df.columns if col.startswith('Studios_')] + \
           [col for col in df.columns if col.startswith('Source_')] + \
           [col for col in df.columns if col.startswith('Genre_')]

X = df[features]
y = df['Score']

# 2. Train the Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Predict Anime Score from User Input
def predict_anime_score(model, X_columns):
    input_data = pd.DataFrame(columns=X_columns)
    input_data.loc[0] = 0

    # User input
    episodes = int(input("Episodes: "))
    year = int(input("Start Year: "))
    anime_type = input("Type (TV, Movie, OVA, etc.): ")
    genres = input("Genres (comma-separated, e.g., Action,Drama): ").split(',')
    studio = input("Studio (e.g., Madhouse): ")

    input_data.at[0, 'Episodes'] = episodes
    input_data.at[0, 'Airing_Year'] = year

    # Type
    type_col = f'Type_{anime_type}'
    if type_col in input_data.columns:
        input_data.at[0, type_col] = 1
    else:
        print(f"Warning: Type '{anime_type}' not found in training data.")

    # Genres
    for genre in genres:
        genre = genre.strip()
        genre_col = f'Genre_{genre}'
        if genre_col in input_data.columns:
            input_data.at[0, genre_col] = 1
        else:
            print(f"Note: Genre '{genre}' not found in model")

    # Studio
    studio_col = f'Studios_{studio}'
    if studio_col in input_data.columns:
        input_data.at[0, studio_col] = 1
    else:
        input_data.at[0, 'Studios_Other'] = 1
        print(f"Note: Studio '{studio}' set as 'Other'")

    # Predict
    predicted_score = model.predict(input_data)[0]
    print(f"\nPredicted Score: {round(predicted_score, 2)}")

# Call the predictor
predict_anime_score(model, X.columns)
