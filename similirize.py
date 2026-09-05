import csv

import pandas as pd

from tfidf import TfIdfVectorizer


def build_corpus(df: pd.DataFrame) -> dict[int, str]:
    return dict(zip(df["movie_id"], df["features"]))


def save_similarities(similarities: dict[int, list[tuple[int, float]]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["movie_id", "similar_id", "similarity"])

        for movie_id, sims in similarities.items():
            for similar_id, score in sims:
                writer.writerow([movie_id, similar_id, round(score, 6)])


if __name__ == "__main__":
    df = pd.read_csv("movies_processed.csv")

    corpus = build_corpus(df)

    vectorizer = TfIdfVectorizer("stop_words.txt")
    vectorizer.fit_transform(corpus)

    similarities = vectorizer.top_n_similar(n=10)
    save_similarities(similarities, "similarities.csv")

    print(similarities[df["movie_id"].iloc[0]])