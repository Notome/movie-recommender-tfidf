import pandas as pd
import ast


credits = pd.read_csv("dataset/credits.csv")
movies = pd.read_csv("dataset/movies.csv")

credits = credits.drop(columns=["title"])

df = credits.merge(
    movies,
    left_on="movie_id",
    right_on="id"
)

def extract_names(value):
    try:
        data = ast.literal_eval(value)

        return " ".join(
            item["name"].replace(" ", "_")
            for item in data
        )

    except:
        return ""


def extract_cast(value, limit=5):
    try:
        data = ast.literal_eval(value)

        return " ".join(
            item["name"].replace(" ", "_")
            for item in data[:limit]
        )

    except:
        return ""


def extract_director(value):
    try:
        data = ast.literal_eval(value)

        for item in data:
            if item["job"] == "Director":
                return item["name"].replace(" ", "_")

        return ""

    except:
        return ""

df["genres"] = df["genres"].apply(extract_names)

df["keywords"] = df["keywords"].apply(extract_names)

df["cast"] = df["cast"].apply(extract_cast)

df["director"] = df["crew"].apply(extract_director)

df["overview"] = df["overview"].fillna("")

df["features"] = (
    df["genres"] + ' ' + df["genres"] + ' ' +
    df["keywords"] + ' ' + df["keywords"] + ' ' + df["keywords"] + ' ' +
    df["cast"] + ' ' + df["cast"] + " " +
    df["director"] + " " + df["director"] + " " + df["director"] + " " + df["director"] + " " +
    df["overview"]
)

df = df[
    [
        "movie_id",
        "title",
        "features"
    ]
]


print(df.head())


df.to_csv(
    "movies_processed.csv",
    index=False
)