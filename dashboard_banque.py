import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import os
import torch
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Orange Bank vs Wave CI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vérification et configuration GPU
def configurer_gpu():
    """Configurer l'utilisation du GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        st.sidebar.success(f"🚀 GPU détecté: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        st.sidebar.warning("❌ GPU non détecté - Utilisation du CPU")
        return device

# Initialisation du modèle Transformers sur GPU
@st.cache_resource
def charger_modele_sentiment_gpu():
    """Charger le modèle de sentiment sur GPU"""
    try:
        device = configurer_gpu()
        
        classificateur = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if torch.cuda.is_available() else -1,  # 0 pour GPU, -1 pour CPU
            batch_size=16,  # Traitement par lots pour plus de vitesse
            truncation=True
        )
        st.success("✅ Modèle Transformers chargé avec succès sur GPU!")
        return classificateur
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        st.info("💡 Installez les dépendances: pip install transformers torch sentencepiece")
        return None

# Liste des stopwords français
STOPWORDS_FR = {
    'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 
    'je', 'la', 'le', 'les', 'leur', 'lui', 'ma', 'me', 'même', 'mes', 'moi', 'mon', 'ne', 'nos', 
    'notre', 'nous', 'on', 'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 
    'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 
    'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées', 'étés', 'étant',
    'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez'
}

def nettoyer_texte(texte):
    """Nettoyer le texte des commentaires"""
    if pd.isna(texte):
        return ""
    texte = str(texte)
    texte = re.sub(r'http\S+', '', texte)
    texte = re.sub(r'@\w+', '', texte)
    texte = re.sub(r'[^\w\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def supprimer_stopwords(texte):
    """Supprimer les stopwords du texte"""
    mots = texte.split()
    mots_filtres = [mot for mot in mots if mot.lower() not in STOPWORDS_FR and len(mot) > 2]
    return ' '.join(mots_filtres)

def analyser_sentiment_batch(commentaires, classificateur):
    """Analyse de sentiment par lots pour plus de vitesse"""
    if not commentaires:
        return [], []
    
    try:
        # Nettoyer et limiter la longueur des textes
        commentaires_clean = [str(comm)[:512] for comm in commentaires if str(comm).strip()]
        
        if not commentaires_clean:
            return [0] * len(commentaires), ['Neutre'] * len(commentaires)
        
        # Analyse par lots
        resultats = classificateur(commentaires_clean)
        
        scores = []
        categories = []
        
        for resultat in resultats:
            label = str(resultat['label']).upper()
            score_confiance = resultat['score']
            
            if 'POSITIVE' in label or '5' in label or '4' in label or 'POSITIF' in label:
                score = score_confiance
                categorie = 'Positif'
            elif 'NEGATIVE' in label or '1' in label or '2' in label or 'NÉGATIF' in label:
                score = -score_confiance
                categorie = 'Négatif'
            else:
                score = 0
                categorie = 'Neutre'
                
            scores.append(score)
            categories.append(categorie)
        
        return scores, categories
        
    except Exception as e:
        st.warning(f"⚠️ Erreur lors de l'analyse par lots: {e}")
        return [0] * len(commentaires), ['Neutre'] * len(commentaires)

def categoriser_themes_banque(texte):
    """Catégoriser les commentaires par thèmes bancaires"""
    if not texte or texte == "":
        return 'Non Classifié'

    texte_lower = texte.lower()

    themes_mapping = {
        'Frais & Commissions': ['frais', 'commission', 'tarif', 'coût', 'prix', 'payant'],
        'Service Client': ['service client', 'support', 'réponse', 'assistance', 'accueil'],
        'Problèmes Techniques': ['bug', 'plantage', 'connexion', 'lenteur', 'erreur'],
        'Paiements & Transferts': ['paiement', 'transfert', 'retrait', 'argent', 'virement'],
        'Sécurité': ['sécurité', 'piratage', 'confiance', 'verrouillage'],
        'Application Mobile': ['application', 'appli', 'interface', 'mobile'],
        'Onboarding': ['inscription', 'vérification', 'document', 'cni'],
        'Fonctionnalités': ['carte', 'chèque', 'notification', 'alerte']
    }

    for theme, mots_cles in themes_mapping.items():
        if any(mot in texte_lower for mot in mots_cles):
            return theme

    return 'Autre'

def calculer_urgence(texte):
    """Calculer un score d'urgence pour les commentaires critiques"""
    if not texte:
        return 0

    texte_lower = texte.lower()
    score_urgence = 0

    mots_urgence = {'arnaque': 3, 'escro': 3, 'vol': 3, 'fraude': 3, 'bloqué': 2, 'urgence': 2}

    for mot, score in mots_urgence.items():
        if mot in texte_lower:
            score_urgence += score

    return min(score_urgence, 5)

def charger_donnees_automatique():
    """Charger automatiquement les fichiers CSV depuis le chemin spécifique"""
    try:
        # Chemin exact vers vos fichiers
        chemin_base = "F:\\IDSI\\Master 2\\Text mining 25-26\\projet_textmining\\data"
        
        # Chemins complets vers vos fichiers
        chemin_orange = os.path.join(chemin_base, "comments_orangeMoney_cleaned.csv")
        chemin_wave = os.path.join(chemin_base, "comments_cleaned_wave.csv")
        
        # Vérifier si les fichiers existent
        if os.path.exists(chemin_orange) and os.path.exists(chemin_wave):
            # Chargement des fichiers
            df_orange = pd.read_csv(chemin_orange)
            df_wave = pd.read_csv(chemin_wave)
            
            return df_orange, df_wave, True
        else:
            # Afficher quels fichiers sont manquants
            if not os.path.exists(chemin_orange):
                st.error(f"❌ Fichier Orange Bank introuvable: {chemin_orange}")
            if not os.path.exists(chemin_wave):
                st.error(f"❌ Fichier Wave introuvable: {chemin_wave}")
            
            return creer_donnees_exemple(), creer_donnees_exemple(), False
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement automatique: {e}")
        return creer_donnees_exemple(), creer_donnees_exemple(), False

def creer_donnees_exemple():
    """Créer des données d'exemple si aucun fichier n'est trouvé"""
    commentaires_orange = [
        "Service client très réactif et professionnel",
        "L'application bug souvent, c'est frustrant",
        "Frais élevés pour les retraits",
        "Transfert d'argent rapide et sécurisé",
        "Je recommande cette banque",
        "Problème de connexion récurrent",
        "Interface utilisateur intuitive",
        "Commission trop élevée sur les virements"
    ]
    
    commentaires_wave = [
        "Gratuit et efficace, parfait pour les transferts",
        "Application simple d'utilisation",
        "Service client difficile à joindre",
        "Très satisfait du service",
        "Problème technique avec l'appli",
        "Rapide et sans frais, excellent",
        "Besoin d'améliorer la sécurité",
        "Support réponse lentement"
    ]
    
    return pd.DataFrame({
        'index': range(len(commentaires_orange)),
        'commentaire': commentaires_orange
    })

@st.cache_data
def preparer_donnees_gpu(_df_orange, _df_wave, _classificateur):
    """Préparer les données pour l'analyse avec Transformers sur GPU"""
    
    # Standardisation des colonnes
    df_orange = _df_orange.rename(columns={_df_orange.columns[1]: 'Commentaire'})
    df_wave = _df_wave.rename(columns={_df_wave.columns[1]: 'Commentaire'})

    # Ajout de la source
    df_orange['Source'] = 'Orange Bank'
    df_wave['Source'] = 'Wave'

    # Fusion
    df_combined = pd.concat([df_orange, df_wave], ignore_index=True)

    # Nettoyage
    df_combined['Commentaire_Nettoye'] = df_combined['Commentaire'].apply(nettoyer_texte)
    
    # Suppression des stopwords pour le nuage de mots
    df_combined['Commentaire_Sans_Stopwords'] = df_combined['Commentaire_Nettoye'].apply(supprimer_stopwords)

    # Génération de dates simulées
    dates_debut = datetime(2024, 1, 1)
    dates_fin = datetime(2024, 12, 31)

    np.random.seed(42)
    dates_aleatoires = [
        dates_debut + timedelta(days=np.random.randint(0, (dates_fin - dates_debut).days))
        for _ in range(len(df_combined))
    ]

    df_combined['Date'] = dates_aleatoires
    df_combined['Mois'] = df_combined['Date'].dt.to_period('M').astype(str)
    df_combined['JourSemaine'] = df_combined['Date'].dt.day_name()
    df_combined['Heure'] = df_combined['Date'].dt.hour

    # Analyse de sentiment AVEC TRANSFORMERS SUR GPU (PAR LOTS)
    st.info("🧠 Analyse des sentiments avec AI Transformers sur GPU...")
    
    # Préparer les commentaires pour l'analyse par lots
    commentaires_list = df_combined['Commentaire_Nettoye'].tolist()
    
    # Analyser par lots de 32 pour plus de vitesse
    batch_size = 32
    total_batches = (len(commentaires_list) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_scores = []
    all_categories = []
    
    for i in range(0, len(commentaires_list), batch_size):
        batch_end = min(i + batch_size, len(commentaires_list))
        batch_commentaires = commentaires_list[i:batch_end]
        
        status_text.text(f"🔧 Traitement lot {i//batch_size + 1}/{total_batches}...")
        
        scores_batch, categories_batch = analyser_sentiment_batch(batch_commentaires, _classificateur)
        all_scores.extend(scores_batch)
        all_categories.extend(categories_batch)
        
        progress_bar.progress(batch_end / len(commentaires_list))
    
    df_combined['Sentiment'] = all_scores
    df_combined['CategorieSentiment'] = all_categories
    
    progress_bar.empty()
    status_text.empty()

    # Catégorisation des thèmes (vectorisée pour plus de vitesse)
    with st.spinner('Catégorisation des thèmes en cours...'):
        df_combined['ThemePrincipal'] = df_combined['Commentaire_Nettoye'].apply(categoriser_themes_banque)

    # Métriques supplémentaires
    df_combined['LongueurCommentaire'] = df_combined['Commentaire_Nettoye'].str.len()
    df_combined['NbMots'] = df_combined['Commentaire_Nettoye'].str.split().str.len()

    # Score d'urgence
    df_combined['ScoreUrgence'] = df_combined['Commentaire_Nettoye'].apply(calculer_urgence)
    df_combined['CommentaireUrgent'] = df_combined['ScoreUrgence'] >= 3

    st.success("✅ Analyse terminée avec succès!")
    return df_combined

# Fonctions de visualisation (inchangées mais optimisées)
def afficher_kpi(df):
    """Afficher les KPI principaux"""
    st.header("📊 Vue d'Ensemble Comparative")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        volume_orange = len(df[df['Source'] == 'Orange Bank'])
        volume_wave = len(df[df['Source'] == 'Wave'])
        diff_volume = volume_orange - volume_wave
        st.metric(
            "Volume Orange Bank", 
            f"{volume_orange:,}", 
            delta=f"{diff_volume:,}" if diff_volume != 0 else None
        )

    with col2:
        sentiment_orange = df[df['Source'] == 'Orange Bank']['Sentiment'].mean()
        sentiment_wave = df[df['Source'] == 'Wave']['Sentiment'].mean()
        diff_sentiment = sentiment_orange - sentiment_wave
        st.metric(
            "Sentiment Orange Bank", 
            f"{sentiment_orange:.3f}",
            delta=f"{diff_sentiment:.3f}" if diff_sentiment != 0 else None,
            delta_color="inverse" if diff_sentiment < 0 else "normal"
        )

    with col3:
        taux_neg_orange = len(df[(df['Source'] == 'Orange Bank') & (df['CategorieSentiment'] == 'Négatif')]) / len(df[df['Source'] == 'Orange Bank']) * 100
        taux_neg_wave = len(df[(df['Source'] == 'Wave') & (df['CategorieSentiment'] == 'Négatif')]) / len(df[df['Source'] == 'Wave']) * 100
        diff_neg = taux_neg_orange - taux_neg_wave
        st.metric(
            "Taux Négatif Orange", 
            f"{taux_neg_orange:.1f}%",
            delta=f"{diff_neg:.1f}%" if diff_neg != 0 else None,
            delta_color="inverse" if diff_neg > 0 else "normal"
        )

    with col4:
        urgents_orange = len(df[(df['Source'] == 'Orange Bank') & (df['CommentaireUrgent'])])
        urgents_wave = len(df[(df['Source'] == 'Wave') & (df['CommentaireUrgent'])])
        st.metric(
            "Commentaires Urgents Orange", 
            urgents_orange,
            delta=f"{urgents_orange - urgents_wave}" if urgents_orange != urgents_wave else None
        )

def afficher_comparaison_volume_sentiment(df):
    """Afficher la comparaison volume et sentiment"""
    col1, col2 = st.columns(2)

    with col1:
        # Volume par plateforme
        volume_par_source = df['Source'].value_counts().reset_index()
        volume_par_source.columns = ['Source', 'Volume']

        fig_volume = px.bar(
            volume_par_source,
            x='Source',
            y='Volume',
            title="Volume des Commentaires par Plateforme",
            color='Source',
            color_discrete_map={'Orange Bank': '#FF7900', 'Wave': '#00A0E3'}
        )
        st.plotly_chart(fig_volume, use_container_width=True)

    with col2:
        # Sentiment moyen par plateforme
        sentiment_par_source = df.groupby('Source')['Sentiment'].mean().reset_index()

        fig_sentiment = px.bar(
            sentiment_par_source,
            x='Source',
            y='Sentiment',
            title="Sentiment Moyen par Plateforme",
            color='Source',
            color_discrete_map={'Orange Bank': '#FF7900', 'Wave': '#00A0E3'}
        )
        fig_sentiment.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig_sentiment, use_container_width=True)

def afficher_repartition_sentiment(df):
    """Afficher la répartition des sentiments"""
    col1, col2 = st.columns(2)

    with col1:
        # Répartition sentiment Orange Bank
        sentiment_orange = df[df['Source'] == 'Orange Bank']['CategorieSentiment'].value_counts()
        fig_orange = px.pie(
            values=sentiment_orange.values,
            names=sentiment_orange.index,
            title="Orange Bank - Répartition des Sentiments",
            color=sentiment_orange.index,
            color_discrete_map={'Positif': '#4CAF50', 'Neutre': '#FFC107', 'Négatif': '#F44336'}
        )
        st.plotly_chart(fig_orange, use_container_width=True)

    with col2:
        # Répartition sentiment Wave
        sentiment_wave = df[df['Source'] == 'Wave']['CategorieSentiment'].value_counts()
        fig_wave = px.pie(
            values=sentiment_wave.values,
            names=sentiment_wave.index,
            title="Wave - Répartition des Sentiments",
            color=sentiment_wave.index,
            color_discrete_map={'Positif': '#4CAF50', 'Neutre': '#FFC107', 'Négatif': '#F44336'}
        )
        st.plotly_chart(fig_wave, use_container_width=True)

def afficher_analyse_thematique(df):
    """Afficher l'analyse par thèmes"""
    st.header("🎯 Analyse par Thèmes")

    # Distribution des thèmes
    fig_themes = px.sunburst(
        df.explode('ThemePrincipal'),
        path=['Source', 'ThemePrincipal'],
        title="Distribution des Thèmes par Plateforme",
        color='Source',
        color_discrete_map={'Orange Bank': '#FF7900', 'Wave': '#00A0E3'}
    )
    st.plotly_chart(fig_themes, use_container_width=True)

    # Sentiment par thème
    col1, col2 = st.columns(2)

    with col1:
        df_orange = df[df['Source'] == 'Orange Bank']
        sentiment_par_theme_orange = df_orange.groupby('ThemePrincipal')['Sentiment'].mean().sort_values()

        fig_orange_themes = px.bar(
            sentiment_par_theme_orange,
            title="Orange Bank - Sentiment par Thème",
            color=sentiment_par_theme_orange.values,
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig_orange_themes, use_container_width=True)

    with col2:
        df_wave = df[df['Source'] == 'Wave']
        sentiment_par_theme_wave = df_wave.groupby('ThemePrincipal')['Sentiment'].mean().sort_values()

        fig_wave_themes = px.bar(
            sentiment_par_theme_wave,
            title="Wave - Sentiment par Thème",
            color=sentiment_par_theme_wave.values,
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig_wave_themes, use_container_width=True)

def afficher_tendances_temporelles(df):
    """Afficher les tendances temporelles"""
    st.header("📈 Évolution Temporelle")

    # Sentiment sur la période
    sentiment_journalier = df.groupby(['Date', 'Source'])['Sentiment'].mean().reset_index()

    fig_tendance = px.line(
        sentiment_journalier,
        x='Date',
        y='Sentiment',
        color='Source',
        title="Évolution du Sentiment Moyen",
        color_discrete_map={'Orange Bank': '#FF7900', 'Wave': '#00A0E3'}
    )
    fig_tendance.update_yaxes(range=[-1, 1])
    st.plotly_chart(fig_tendance, use_container_width=True)

    # Volume par jour de semaine
    col1, col2 = st.columns(2)

    with col1:
        volume_par_jour = df.groupby(['JourSemaine', 'Source']).size().reset_index(name='Volume')
        jours_ordre = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        volume_par_jour['JourSemaine'] = pd.Categorical(volume_par_jour['JourSemaine'], categories=jours_ordre, ordered=True)
        volume_par_jour = volume_par_jour.sort_values('JourSemaine')

        fig_volume_jour = px.bar(
            volume_par_jour,
            x='JourSemaine',
            y='Volume',
            color='Source',
            barmode='group',
            title="Volume des Commentaires par Jour de Semaine",
            color_discrete_map={'Orange Bank': '#FF7900', 'Wave': '#00A0E3'}
        )
        st.plotly_chart(fig_volume_jour, use_container_width=True)

    with col2:
        # Heatmap des heures d'activité
        activite_heure = df.groupby(['Heure', 'Source']).size().reset_index(name='Volume')

        fig_heatmap = px.density_heatmap(
            activite_heure,
            x='Heure',
            y='Source',
            z='Volume',
            title="Activité par Heure de la Journée",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def afficher_commentaires_critiques(df):
    """Afficher les commentaires critiques"""
    st.header("🚨 Commentaires Critiques et Urgents")

    tab1, tab2 = st.tabs(["Commentaires Urgents", "Commentaires les plus Négatifs"])

    with tab1:
        st.subheader("Commentaires Requérant une Action Immédiate")
        commentaires_urgents = df[df['CommentaireUrgent']].sort_values('ScoreUrgence', ascending=False)

        for idx, row in commentaires_urgents.head(10).iterrows():
            with st.expander(f"🚨 {row['Source']} - Urgence: {row['ScoreUrgence']}/5 - Thème: {row['ThemePrincipal']}"):
                st.write(f"**Commentaire:** {row['Commentaire']}")
                st.write(f"**Score de sentiment:** {row['Sentiment']:.3f}")
                st.write(f"**Catégorie:** {row['CategorieSentiment']}")
                st.write(f"**Date:** {row['Date'].strftime('%Y-%m-%d')}")

    with tab2:
        st.subheader("Commentaires les plus Négatifs")
        commentaires_negatifs = df[df['CategorieSentiment'] == 'Négatif'].nsmallest(10, 'Sentiment')

        for idx, row in commentaires_negatifs.iterrows():
            with st.expander(f"🔴 {row['Source']} - Score: {row['Sentiment']:.3f} - Thème: {row['ThemePrincipal']}"):
                st.write(f"**Commentaire:** {row['Commentaire']}")
                st.write(f"**Confiance AI:** {abs(row['Sentiment']):.1%}")
                st.write(f"**Longueur:** {row['LongueurCommentaire']} caractères")
                st.write(f"**Date:** {row['Date'].strftime('%Y-%m-%d')}")

def afficher_mots_cles(df):
    """Afficher l'analyse des mots-clés avec stopwords supprimés"""
    st.header("🔍 Analyse des Mots-Clés (sans stopwords)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orange Bank - Mots Fréquents")
        text_orange = ' '.join(df[df['Source'] == 'Orange Bank']['Commentaire_Sans_Stopwords'].dropna())

        if text_orange.strip():
            wordcloud_orange = WordCloud(
                width=500, 
                height=350, 
                background_color='white',
                colormap='Oranges',
                stopwords=STOPWORDS_FR,
                max_words=100,
                relative_scaling=0.5
            ).generate(text_orange)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud_orange, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Orange Bank - Mots les plus fréquents (sans stopwords)', fontsize=16, pad=20)
            st.pyplot(fig)
            
            # Statistiques des mots
            mots_orange = text_orange.split()
            st.write(f"**{len(mots_orange)} mots analysés** (stopwords exclus)")
        else:
            st.info("Aucun commentaire disponible pour Orange Bank")

    with col2:
        st.subheader("Wave - Mots Fréquents")
        text_wave = ' '.join(df[df['Source'] == 'Wave']['Commentaire_Sans_Stopwords'].dropna())

        if text_wave.strip():
            wordcloud_wave = WordCloud(
                width=500, 
                height=350, 
                background_color='white',
                colormap='Blues',
                stopwords=STOPWORDS_FR,
                max_words=100,
                relative_scaling=0.5
            ).generate(text_wave)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud_wave, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Wave - Mots les plus fréquents (sans stopwords)', fontsize=16, pad=20)
            st.pyplot(fig)
            
            # Statistiques des mots
            mots_wave = text_wave.split()
            st.write(f"**{len(mots_wave)} mots analysés** (stopwords exclus)")
        else:
            st.info("Aucun commentaire disponible pour Wave")

# Application principale
def main():
    st.title("🏦 Analyse des Commentaires - Orange Bank CI vs Wave CI")
    st.markdown("### 🤗 Powered by Transformers AI - Optimisé GPU 🚀")

    # Configuration GPU
    device = configurer_gpu()

    # Chargement du modèle sur GPU
    with st.spinner("🔄 Chargement du modèle AI Transformers sur GPU..."):
        classificateur = charger_modele_sentiment_gpu()
    
    if classificateur is None:
        st.error("""
        ❌ Impossible de charger le modèle AI. 
        
        **Veuillez installer les dépendances :**
        ```bash
        pip install transformers torch sentencepiece
        ```
        """)
        return

    # Chargement automatique des données
    st.sidebar.header("📁 Chargement des Données")
    
    with st.spinner('Chargement automatique des données...'):
        df_orange, df_wave, donnees_reelles = charger_donnees_automatique()
    
    if donnees_reelles:
        st.success("✅ Données chargées automatiquement!")
    else:
        st.warning("⚠️ Utilisation des données d'exemple")

    # Préparation des données AVEC TRANSFORMERS SUR GPU
    if st.button("🚀 Lancer l'analyse AI sur GPU", type="primary"):
        df_combined = preparer_donnees_gpu(df_orange, df_wave, classificateur)
        
        # Stocker les données dans la session pour les filtres rapides
        st.session_state.df_combined = df_combined
        st.session_state.analysis_done = True
        st.rerun()

    # Si l'analyse est déjà faite, afficher les résultats
    if hasattr(st.session_state, 'analysis_done') and st.session_state.analysis_done:
        df_combined = st.session_state.df_combined
        
        # Filtres
        st.sidebar.header("🎛️ Filtres")

        sources_selectionnees = st.sidebar.multiselect(
            "Plateformes",
            options=['Orange Bank', 'Wave'],
            default=['Orange Bank', 'Wave']
        )

        themes_selectionnes = st.sidebar.multiselect(
            "Thèmes",
            options=df_combined['ThemePrincipal'].unique(),
            default=df_combined['ThemePrincipal'].unique()
        )

        sentiments_selectionnes = st.sidebar.multiselect(
            "Sentiments",
            options=['Positif', 'Neutre', 'Négatif'],
            default=['Positif', 'Neutre', 'Négatif']
        )

        # Application des filtres (INSTANTANÉ maintenant)
        df_filtre = df_combined[
            (df_combined['Source'].isin(sources_selectionnees)) &
            (df_combined['ThemePrincipal'].isin(themes_selectionnes)) &
            (df_combined['CategorieSentiment'].isin(sentiments_selectionnes))
        ]

        # Navigation
        st.sidebar.header("📊 Navigation")
        page = st.sidebar.radio(
            "Sélectionner une page:",
            ["Vue d'Ensemble", "Analyse Thématique", "Tendances Temporelles", "Commentaires Critiques", "Mots-Clés"]
        )

        # Affichage des pages (INSTANTANÉ maintenant)
        if page == "Vue d'Ensemble":
            afficher_kpi(df_filtre)
            afficher_comparaison_volume_sentiment(df_filtre)
            afficher_repartition_sentiment(df_filtre)

        elif page == "Analyse Thématique":
            afficher_analyse_thematique(df_filtre)

        elif page == "Tendances Temporelles":
            afficher_tendances_temporelles(df_filtre)

        elif page == "Commentaires Critiques":
            afficher_commentaires_critiques(df_filtre)

        elif page == "Mots-Clés":
            afficher_mots_cles(df_filtre)

        # Statistiques globales dans la sidebar
        st.sidebar.header("📈 Statistiques Globales")
        st.sidebar.metric("Total Commentaires", len(df_filtre))
        st.sidebar.metric("Sentiment Moyen", f"{df_filtre['Sentiment'].mean():.3f}")
        
        taux_positivite = (len(df_filtre[df_filtre['CategorieSentiment'] == 'Positif']) / len(df_filtre) * 100) if len(df_filtre) > 0 else 0
        st.sidebar.metric("Taux de Positivité", f"{taux_positivite:.1f}%")

        # Téléchargement des données analysées
        st.sidebar.header("💾 Export")
        csv = df_combined.to_csv(index=False, encoding='utf-8-sig')
        st.sidebar.download_button(
            label="📥 Télécharger les données analysées",
            data=csv,
            file_name="commentaires_analyse_ai_gpu.csv",
            mime="text/csv"
        )

    else:
        st.info("👆 Cliquez sur le bouton 'Lancer l'analyse AI sur GPU' pour commencer l'analyse")

if __name__ == "__main__":
    main()