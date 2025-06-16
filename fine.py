import json

# Load your raw movie data (replace this with your actual JSON file)
with open("sample_movies.json", "r") as f:
    movies = json.load(f)

train_data = []

for movie in movies:
    title = movie.get("title", "Unknown")
    descriptions = movie.get("description", {})
    plot = descriptions.get("tv", [""])[0]  # fallback to tv
    genre = movie.get("genres", ["drama"])[0]
    language = movie.get("languages", ["a regional"])[0]

    if plot.lower() == "movie." or not plot.strip():
        continue  # skip placeholder entries

    entry = {
        "instruction": f"Write a short, spoiler-free and vivid description of a {language} {genre} film.",
        "input": f"Masked plot of {title}:\n{plot}",
        "output": plot  # Here, use the same plot as target since we don't have a better reference
    }

    train_data.append(entry)

# Save as JSONL
with open("movie_descriptions.jsonl", "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")
