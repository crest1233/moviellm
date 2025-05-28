import spacy
import wikipediaapi
import json
import re
import subprocess
import mysql.connector
from mysql.connector import errorcode

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Setup Wikipedia API client
wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MovieDescBot/1.0 (tanishk.singh@example.com)'
)

def clean_and_mask_text(text, starcast):
    doc = nlp(text)
    masked_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            masked_text = re.sub(rf'\b{re.escape(ent.text)}\b', '[REDACTED]', masked_text, flags=re.IGNORECASE)
    for actor in starcast:
        masked_text = re.sub(rf'\b{re.escape(actor)}\b', '[REDACTED]', masked_text, flags=re.IGNORECASE)
    return masked_text

def get_wikipedia_context(movie, sentences=2):
    search_terms = []
    for genre in movie.get("genres", []):
        search_terms.append(f"{genre} film")
    search_terms.extend([
        "Indian cinema", "Bollywood", "family drama in India", "Indian military",
        "British Raj", "Indian rural life", "crime in India", "surrogacy in India"
    ])
    for term in search_terms:
        page = wiki.page(term)
        if page.exists():
            return '. '.join(page.summary.split('. ')[:sentences]) + '.'
    return "No context available."

def construct_prompt(movie):
    merged_description = " ".join(
        movie["description"].get("cinema", []) +
        movie["description"].get("tvplus", [])
    )
    plot_outline = movie.get("plot outline", "")
    full_text = f"{merged_description} {plot_outline}"
    masked_text = clean_and_mask_text(full_text, movie.get("starcast", []))
    genre_context = get_wikipedia_context(movie)

    prompt = f"""
Write a short, spoiler-free and vivid description of a {movie['languages'][0]} {movie['genres'][0]} film.
Avoid mentioning the title or any actor names.

Plot:
{masked_text}

Background Context:
{genre_context}

Create a description that sparks curiosity and reflects the genre's tone.
"""
    return prompt.strip()

def run_llama(prompt):
    process = subprocess.Popen(
        ['ollama', 'run', 'llama3.2'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    output, error = process.communicate(input=prompt.encode())
    return output.decode().strip()

def ensure_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS movie_descriptions (
        id INT AUTO_INCREMENT PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        description TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB;
    """)

def main():
    # Load your movies JSON
    with open("sample_movies.json", "r") as f:
        movies = json.load(f)

    # Connect to MySQL (auto-creates database if missing)
    try:
        cnx = mysql.connector.connect(
            host="localhost",
            user="root",
            password="tanishk07",
            database="movie"
        )
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            cnx = mysql.connector.connect(
                host="localhost",
                user="root",
                password="tanishk07"
            )
            cursor = cnx.cursor()
            cursor.execute(
                "CREATE DATABASE movie CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
            cnx.database = "movie"
        else:
            print("MySQL connection error:", err)
            return

    cursor = cnx.cursor()
    ensure_table(cursor)

    # Loop through each movie, generate and store description
    for movie in movies:
        title = movie["title"]
        prompt = construct_prompt(movie)
        description = run_llama(prompt)

        print(f"Storing description for '{title}'…")
        insert_sql = """
            INSERT INTO movie_descriptions (title, description)
            VALUES (%s, %s)
        """
        cursor.execute(insert_sql, (title, description))
        cnx.commit()
        print("✔️ Stored.\n")

    cursor.close()
    cnx.close()

if __name__ == "__main__":
    main()
