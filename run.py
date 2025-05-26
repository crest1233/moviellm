import spacy
import wikipediaapi
import json
import re
import subprocess

nlp = spacy.load("en_core_web_sm")

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
    merged_description = " ".join(movie["description"].get("cinema", []) + movie["description"].get("tvplus", []))
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

Create a teaser that sparks curiosity and reflects the genre's tone.
"""
    return prompt.strip()

def run_llama(prompt):
    process = subprocess.Popen(['ollama', 'run', 'llama3.2'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    output, error = process.communicate(input=prompt.encode())
    return output.decode().strip()

def main():
    with open("sample_movies.json", "r") as f:
        movies = json.load(f)

    for movie in movies:
        print("\n" + "=" * 60)
        print(f"Generating description for: {movie['title']}")
        print("=" * 60)
        prompt = construct_prompt(movie)
        print("Prompt sent to LLaMA:\n", prompt)
        print("\nGenerated Output:\n")
        output = run_llama(prompt)
        print(output)

if __name__ == "__main__":
    main()
