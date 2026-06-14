# 🎮 GameRAG – System rekomendacji gier wideo

Projekt zaliczeniowy z przedmiotu Sztuczna Inteligencja.  
System RAG oparty na lokalnym modelu AI (Ollama) i własnej bazie 100 gier wideo.

## Architektura

```
games.csv  →  ChromaDB (embeddingi)  →  Retrieval
                                             ↓
Zapytanie użytkownika  ──────────────→  LLM (llama3.2)  →  Odpowiedź
```

## Wymagania

- Python 3.10+
- [Ollama](https://ollama.com/download/windows) zainstalowane i uruchomione

## Instalacja

```bash
# 1. Pobierz modele Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Utwórz środowisko wirtualne
python -m venv venv --without-pip
venv\Scripts\activate
python -m ensurepip --upgrade

# 3. Zainstaluj zależności
pip install -r requirements.txt

# 4. Uruchom aplikację
streamlit run app.py
```

## Użycie

1. Otwórz `http://localhost:8501` w przeglądarce
2. Wpisz zapytanie, np.:
   - *"polecisz coś w klimacie survival horror?"*
   - *"gra RPG z trudną walką i mrocznym fantasy"*
   - *"coś spokojnego do relaksu z farminggiem"*
3. Kliknij **Szukaj**
4. System znajdzie pasujące gry i wygeneruje rekomendację AI

## Stack technologiczny

| Komponent | Narzędzie |
|---|---|
| LLM | Ollama + llama3.2 |
| Embeddingi | nomic-embed-text |
| Baza wektorowa | ChromaDB (lokalna) |
| UI | Streamlit |
| Dane | CSV (100 gier) |

## Struktura projektu

```
projekt_rag/
├── app.py           # Główna aplikacja Streamlit
├── games.csv        # Baza 100 gier
├── requirements.txt # Zależności
├── chroma_db/       # Baza wektorowa (generowana automatycznie)
└── README.md
```
