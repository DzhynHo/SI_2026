# Sztuczna Inteligencja - Laboratoria

Repozytorium zawiera zadania zrealizowane podczas kursu Sztucznej Inteligencji (ANS). Wszystkie projekty zostały wykonane w środowisku Google Colab przy użyciu biblioteki TensorFlow oraz wybranych modeli LLM (Groq API, Ollama).

## Autor

* **Imię i Nazwisko:** Khylchenko Valeriia

## Spis projektów

### 1. Model Liniowy (Lab 1 & 2)
* Prosty algorytm uczenia maszynowego "od zera" i jego wersja w TensorFlow/Keras.

### 2. Rozpoznawanie pisma odręcznego (Lab 3)
* Sieć neuronowa (MLP) klasyfikująca cyfry ze zbioru **MNIST**.
* Test modelu na własnych zdjęciach cyfr (`OpenCV`).

### 3. Klasyfikacja - Audiobooks (Lab 4a & 4b)
* Preprocessing danych klienckich i budowa modelu klasyfikacyjnego przewidującego powrót klienta (dokładność ~79%).

### 4. Predykcja cen akcji - LSTM/GRU (Lab 5)
* Predykcja ceny akcji IBM sieciami **LSTM** i **GRU**.
* Porównanie optymalizatorów, funkcji straty i konfiguracji modelu.

### 5. LLM - podsumowanie stron internetowych (Lab 6)
* Web scraper + podsumowanie treści strony przez model LLM (Groq API) w języku polskim.
* Porównanie jakości trzech modeli.

### 6. RAG - Retrieval-Augmented Generation (Lab 7)
* System odpowiadający na pytania na podstawie własnej bazy PDF (FAISS + embeddingi).
* Model LLM (Ollama, lokalnie na GPU) generuje odpowiedzi tylko na podstawie kontekstu, podając źródła.

## Technologie

* Python 3.x
* TensorFlow / Keras
* NumPy, Matplotlib, Pandas, Scikit-learn
* OpenCV
* Groq API, Ollama
* FAISS, Sentence-Transformers, PyMuPDF
