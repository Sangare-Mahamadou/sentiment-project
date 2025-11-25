# 🚀 Sentiment Analysis & Dashboard Project

Bienvenue dans ce projet d’analyse intelligente des commentaires Facebook pour Orange Money CI et Wave CI ! 🔥

---

## 🤩 Fonctionnalités clés

- **Scraping automatisé** des commentaires Facebook avec Selenium
- **Nettoyage avancé**, text mining, clustering et analyse de sentiments via spaCy, scikit-learn et Hugging Face
- **Visualisation interactive** des résultats à travers des dashboards personnalisés (Streamlit)
- **Exploration multi-sources** : Orange Money CI, Wave CI & banque
- **Visualisation** : en utilisant Streamlit
---

## 🗂️ Architecture du projet

sentiment-project/
│
├── data/
│   ├── comments.csv
│   ├── comments_cleaned.csv
│   ├── comments_cleaned_wave.csv
│   ├── comments_combined_cleaned_sentiment.csv
│   ├── comments_orangeMoney_cleaned.csv
│   ├── comments_wave.csv
│   ├── final_database.csv
│   └── facebook.json       # À créer à partir de vos cookies Facebook !
│
├── notebooks/
│   ├── scraping.ipynb
│   ├── text_mining.ipynb
│   └── y.ipynb
│
├── dashboard.py
├── dashboard_banque.py
├── requirements.txt
├── README.md


---

## ⚠️ Préparation des cookies Facebook

Pour pouvoir scraper les commentaires Facebook, **vous devez télécharger vos cookies Facebook** après connexion dans votre navigateur.  
Sauvegardez le fichier au format `.json` en le nommant **facebook.json**, puis placez-le dans le dossier `data/` du projet.

> 🔍 Cherchez sur Google : « Comment exporter ses cookies Facebook en .json »  ou via l'extension `J2TEAM Cookies`
> 📁 Fichier attendu : `data/facebook.json`

---

## 🛠️ Prérequis

- Python 3.8+
- **Librairies :**
    - spacy
    - scikit-learn
    - streamlit
    - transformers
    - selenium
    - pandas
    - numpy

---

## ⚡ Installation

git clone https://github.com/TON-USERNAME/sentiment-project.git
cd sentiment-project
pip install -r requirements.txt


---

## 🚀 Utilisation

1. **Placez tous vos fichiers CSV/JSON dans** `data/`
2. **Créez et ajoutez vos cookies Facebook** dans `data/facebook.json`
3. **Explorez et analysez les données dans les notebooks :**
    - `notebooks/scraping.ipynb` : Récupération des commentaires Facebook
    - `notebooks/text_mining.ipynb` : Prétraitement, clustering & sentiment
4. **Visualisez les résultats avec les dashboards :**
    ```
    streamlit run dashboard.py
    streamlit run dashboard_banque.py
    ```

---

### Prêt à révéler les tendances et les émotions dans vos données Facebook ? Installez, chargez vos cookies, et lancez l’analyse !


