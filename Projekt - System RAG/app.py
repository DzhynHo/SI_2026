import streamlit as st
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import ollama
import csv as _csv
import re as _re

# ── Konfiguracja ──────────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "llama3.2"
CSV_PATH    = "games.csv"
DB_PATH     = "./chroma_db"
COLLECTION  = "games"

st.set_page_config(page_title="GameRAG", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F0EFEA;
    color: #0A0A0A;
}

/* Ukryj domyślny header Streamlit */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Główny kontener */
.main .block-container {
    padding: 3rem 4rem;
    max-width: 900px;
}

/* Tytuł */
h1 {
    font-size: 4rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.0 !important;
    color: #0A0A0A !important;
    text-transform: uppercase;
    margin-bottom: 0.5rem !important;
}

h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #0A0A0A !important;
}

/* Niebieski akcent – linia pod tytułem */
h1::after {
    content: '';
    display: block;
    width: 80px;
    height: 6px;
    background: #0047FF;
    margin-top: 0.6rem;
    border-radius: 3px;
}

/* Separator */
hr {
    border: none;
    border-top: 2px solid #0A0A0A;
    margin: 2rem 0;
}

/* Input */
input[type="text"], textarea {
    background: #FFFFFF !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 0px !important;
    font-size: 1rem !important;
    color: #0A0A0A !important;
    padding: 0.75rem 1rem !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #0047FF !important;
    box-shadow: 0 0 0 2px rgba(0,71,255,0.15) !important;
}

/* Przycisk główny */
button[kind="primary"], .stButton > button {
    background-color: #0047FF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: background 0.15s;
}
.stButton > button:hover {
    background-color: #0A0A0A !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: #0047FF !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    background: #FFFFFF !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
}
.streamlit-expanderContent {
    border: 2px solid #0A0A0A !important;
    border-top: none !important;
    background: #FAFAFA !important;
}

/* Info / Success boxy */
.stAlert {
    border-radius: 0 !important;
    border-left: 5px solid #0047FF !important;
    background: #FFFFFF !important;
    color: #0A0A0A !important;
}

/* Caption */
.stCaption {
    color: #555 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase;
}

/* Spinner */
.stSpinner > div { border-top-color: #0047FF !important; }
</style>
""", unsafe_allow_html=True)

def wczytaj_csv():
    """Wczytuje plik CSV z grami i zwraca DataFrame."""
    rows = []
    with open(CSV_PATH, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = _csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) >= 4:
                rows.append(row[:4])
            elif len(row) == 1:
                m = _re.match(r'^(.+?),(\d{4}),([^,]+),"?(.*?)"?\s*$', row[0], _re.DOTALL)
                if m:
                    rows.append([m.group(1).strip(), m.group(2).strip(),
                                 m.group(3).strip(), m.group(4).strip()])
    df = pd.DataFrame(rows, columns=["nazwa", "rok", "gatunek", "opis"])
    df["rok"] = pd.to_numeric(df["rok"], errors="coerce")
    return df

@st.cache_resource

def zaladuj_baze():
    """Tworzy bazę wektorową ChromaDB i indeksuje gry (tylko raz)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )
    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=ef)

    if collection.count() == 0:
        df = wczytaj_csv()
        with st.spinner("Indeksuję gry w ChromaDB..."):
            docs, ids, metas = [], [], []
            for i, row in df.iterrows():
                tekst = f"{row['nazwa']} ({row['rok']}) – {row['gatunek']}\n{row['opis']}"
                docs.append(tekst)
                ids.append(str(i))
                metas.append({
                    "nazwa":   str(row["nazwa"]),
                    "rok":     int(row["rok"]) if pd.notna(row["rok"]) else 0,
                    "gatunek": str(row["gatunek"]),
                })
            collection.add(documents=docs, ids=ids, metadatas=metas)
    return collection


def szukaj_w_bazie(collection, zapytanie, n=5):
    """Wyszukuje n gier najbardziej podobnych do zapytania (wyszukiwanie semantyczne)."""
    wyniki = collection.query(query_texts=[zapytanie], n_results=n)
    return wyniki["documents"][0], wyniki["metadatas"][0]

def generuj_odpowiedz(zapytanie, kontekst):
    """Wysyła zapytanie + kontekst do LLM i zwraca rekomendację."""
    blok = "\n\n".join(f"-  {d}" for d in kontekst)
    prompt = f"""Jesteś ekspertem od gier wideo. Użytkownik zadał pytanie:
"{zapytanie}"

Na podstawie poniższych gier z bazy danych udziel pomocnej odpowiedzi po polsku.
Poleć najlepiej pasujące tytuły i krótko wyjaśnij dlaczego pasują.

KONTEKST Z BAZY:
{blok}

Odpowiedź:"""
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


st.title("GameRAG")
st.markdown(
    "<p style='font-size:1.1rem; color:#444; margin-top:-0.5rem;'>"
    "System rekomendacji gier — RAG · ChromaDB · LLM"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

collection = zaladuj_baze()

zapytanie = st.text_input(
    "Wpisz czego szukasz:",
    placeholder="np. trudna gra z mrocznym klimatem fantasy"
)
n_wynikow = st.slider("Liczba wyników", min_value=1, max_value=10, value=5)

if st.button("SZUKAJ", type="primary") and zapytanie:

    st.markdown("---")
    st.subheader("Krok 1 – Zapytanie użytkownika")
    st.info(f'"{zapytanie}"')

    st.subheader("Krok 2 – Wyszukiwanie semantyczne w bazie wektorowej")
    st.caption(f"Model embeddingów: **{EMBED_MODEL}** | Baza wektorowa: **ChromaDB**")

    with st.spinner("Szukam podobnych gier..."):
        docs, metas = szukaj_w_bazie(collection, zapytanie, n=n_wynikow)

    st.write(f"Znaleziono **{len(docs)}** najbardziej podobnych gier:")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        with st.expander(
            f"#{i+1}  {meta.get('nazwa', '?')}  "
            f"({meta.get('rok', '?')})  –  {meta.get('gatunek', '?')}"
        ):
            st.write(doc)

    st.subheader("Krok 3 – Generowanie odpowiedzi przez LLM")
    st.caption(f"Model językowy: **{LLM_MODEL}** (Ollama)")

    with st.spinner("Generuję odpowiedź na podstawie znalezionych gier..."):
        odpowiedz = generuj_odpowiedz(zapytanie, docs)

    st.success(odpowiedz)


#Stopka
st.markdown("---")
st.caption("System RAG | Projekt SI | Akademia Nauk Stosowanych w Elblągu")
