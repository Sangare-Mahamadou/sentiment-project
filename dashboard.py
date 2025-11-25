
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from itertools import combinations

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Analyse Sentiments",
    page_icon="📊",
    layout="wide"
)

# Titre principal
st.title("📊 Dashboard d'Analyse des Sentiments")
st.markdown("### Orange Money vs WAVE")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('final_database.csv')
    return df

try:
    df = load_data()
    
    # Sidebar - Filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par service
    services = st.sidebar.multiselect(
        "Sélectionner les services",
        options=df['Service'].unique(),
        default=df['Service'].unique()
    )
    
    # Filtre par sentiment
    sentiments = st.sidebar.multiselect(
        "Sélectionner les sentiments",
        options=df['sentiment'].unique(),
        default=df['sentiment'].unique()
    )
    
    # Filtre par cluster
    clusters = st.sidebar.multiselect(
        "Sélectionner les clusters",
        options=sorted(df['cluster'].unique()),
        default=sorted(df['cluster'].unique())
    )
    
    # Application des filtres
    df_filtered = df[
        (df['Service'].isin(services)) & 
        (df['sentiment'].isin(sentiments)) &
        (df['cluster'].isin(clusters))
    ]
    
    # Métriques principales
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Commentaires", len(df_filtered))
    with col2:
        st.metric("😊 Positifs", len(df_filtered[df_filtered['sentiment'] == 'POSITIVE']))
    with col3:
        st.metric("😐 Neutres", len(df_filtered[df_filtered['sentiment'] == 'NEUTRAL']))
    with col4:
        st.metric("😞 Négatifs", len(df_filtered[df_filtered['sentiment'] == 'NEGATIVE']))
    
    st.markdown("---")
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des Sentiments par Service")
        sentiment_counts = df_filtered.groupby(['Service', 'sentiment']).size().reset_index(name='count')
        fig1 = px.bar(
            sentiment_counts,
            x='Service',
            y='count',
            color='sentiment',
            barmode='group',
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEUTRAL': '#3498db',
                'NEGATIVE': '#e74c3c'
            },
            title="Répartition des sentiments"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Proportion des Sentiments")
        sentiment_pie = df_filtered['sentiment'].value_counts()
        fig2 = px.pie(
            values=sentiment_pie.values,
            names=sentiment_pie.index,
            color=sentiment_pie.index,
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEUTRAL': '#3498db',
                'NEGATIVE': '#e74c3c'
            },
            title="Distribution globale"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Deuxième ligne de graphiques
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Distribution des Clusters")
        cluster_counts = df_filtered['cluster'].value_counts().sort_index()
        fig3 = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Nombre de commentaires'},
            title="Commentaires par cluster",
            color=cluster_counts.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        st.subheader("Distribution du Nombre de Mots")
        fig4 = px.histogram(
            df_filtered,
            x='nb_mots',
            nbins=30,
            title="Longueur des commentaires",
            labels={'nb_mots': 'Nombre de mots', 'count': 'Fréquence'},
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Visualisation 2D des clusters
    st.markdown("---")
    st.subheader("Visualisation 2D des Clusters (Coordonnées x, y)")
    
    fig5 = px.scatter(
        df_filtered,
        x='x',
        y='y',
        color='sentiment',
        symbol='Service',
        hover_data=['clean_sentence', 'cluster', 'nb_mots'],
        title="Projection 2D des commentaires",
        color_discrete_map={
            'POSITIVE': '#2ecc71',
            'NEUTRAL': '#3498db',
            'NEGATIVE': '#e74c3c'
        },
        size='nb_mots',
        size_max=15
    )
    fig5.update_layout(height=600)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Comparaison entre services
    st.markdown("---")
    st.subheader("📊 Comparaison Orange Money vs WAVE")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Statistiques par service
        stats_by_service = df_filtered.groupby('Service').agg({
            'nb_mots': 'mean',
            'sentiment': lambda x: (x == 'POSITIVE').sum() / len(x) * 100
        }).round(2)
        stats_by_service.columns = ['Moy. Mots', 'Taux Positif (%)']
        st.dataframe(stats_by_service, use_container_width=True)
    
    with col6:
        # Sentiment score par service
        sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        df_filtered['sentiment_score'] = df_filtered['sentiment'].map(sentiment_mapping)
        avg_sentiment = df_filtered.groupby('Service')['sentiment_score'].mean().reset_index()
        
        fig6 = px.bar(
            avg_sentiment,
            x='Service',
            y='sentiment_score',
            title="Score moyen de sentiment (-1 à 1)",
            color='sentiment_score',
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    # Top mots les plus fréquents
    st.markdown("---")
    st.subheader("🔤 Mots les Plus Fréquents")
    
    col7, col8 = st.columns(2)
    
    for idx, service in enumerate(services):
        df_service = df_filtered[df_filtered['Service'] == service]
        
        # Extraction des mots
        all_words = ' '.join(df_service['clean_sentence'].dropna()).split()
        word_freq = Counter(all_words).most_common(15)
        
        if word_freq:
            words_df = pd.DataFrame(word_freq, columns=['Mot', 'Fréquence'])
            
            fig_words = px.bar(
                words_df,
                x='Fréquence',
                y='Mot',
                orientation='h',
                title=f"Top 15 mots - {service}",
                color='Fréquence',
                color_continuous_scale='blues'
            )
            
            if idx % 2 == 0:
                col7.plotly_chart(fig_words, use_container_width=True)
            else:
                col8.plotly_chart(fig_words, use_container_width=True)
    
    # ============= ANALYSE THÉMATIQUE =============
    st.markdown("---")
    st.header("🎯 Analyse Thématique")
    
    # Définition des thèmes avec mots-clés
    themes = {
        'Réseau & Connexion': ['réseau', 'connexion', 'internet', 'signal', 'couverture', 'débit', 'ligne'],
        'Transaction & Paiement': ['paiement', 'transaction', 'transfert', 'argent', 'envoyer', 'recevoir', 'retrait', 'depot'],
        'Service Client': ['service', 'client', 'support', 'assistance', 'aide', 'réponse', 'appel', 'contact'],
        'Frais & Tarification': ['frais', 'coût', 'prix', 'tarif', 'cher', 'gratuit', 'promotion', 'promo', 'mega'],
        'Application & Interface': ['application', 'app', 'interface', 'bug', 'erreur', 'probleme', 'fonctionner'],
        'Rapidité & Efficacité': ['rapide', 'lent', 'vitesse', 'instant', 'temps', 'attendre', 'délai'],
        'Sécurité': ['sécurité', 'securite', 'fraude', 'arnaque', 'fiable', 'confiance', 'sûr'],
        'Disponibilité': ['disponible', 'indisponible', 'accessible', 'marche', 'fonctionne', 'down']
    }
    
    # Fonction pour détecter les thèmes
    def detect_themes(text, themes_dict):
        detected = []
        text_lower = str(text).lower()
        for theme, keywords in themes_dict.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(theme)
        return detected if detected else ['Autre']
    
    # Application de la détection thématique
    df_filtered['themes'] = df_filtered['clean_sentence'].apply(lambda x: detect_themes(x, themes))
    
    # Exploser les thèmes pour l'analyse (un commentaire peut avoir plusieurs thèmes)
    df_exploded = df_filtered.explode('themes')
    
    # Vue d'ensemble des thèmes
    st.subheader("📊 Vue d'ensemble des thèmes")
    
    col_theme1, col_theme2 = st.columns(2)
    
    with col_theme1:
        # Distribution globale des thèmes
        theme_counts = df_exploded['themes'].value_counts()
        fig_themes_global = px.bar(
            x=theme_counts.values,
            y=theme_counts.index,
            orientation='h',
            title="Distribution globale des thèmes",
            labels={'x': 'Nombre de mentions', 'y': 'Thème'},
            color=theme_counts.values,
            color_continuous_scale='teal'
        )
        fig_themes_global.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_themes_global, use_container_width=True)
    
    with col_theme2:
        # Thèmes par service
        theme_service = df_exploded.groupby(['Service', 'themes']).size().reset_index(name='count')
        fig_themes_service = px.bar(
            theme_service,
            x='themes',
            y='count',
            color='Service',
            barmode='group',
            title="Thèmes par service",
            labels={'themes': 'Thème', 'count': 'Nombre de mentions'},
            color_discrete_sequence=['#FF6B35', '#004E89']
        )
        fig_themes_service.update_xaxes(tickangle=45)
        fig_themes_service.update_layout(height=500)
        st.plotly_chart(fig_themes_service, use_container_width=True)
    
    # Analyse sentiment par thème
    st.subheader("💭 Sentiment par thème")
    
    col_sent1, col_sent2 = st.columns(2)
    
    with col_sent1:
        # Heatmap sentiment x thème
        sentiment_theme = pd.crosstab(df_exploded['themes'], df_exploded['sentiment'])
        sentiment_theme_pct = sentiment_theme.div(sentiment_theme.sum(axis=1), axis=0) * 100
        
        fig_heatmap = px.imshow(
            sentiment_theme_pct.T,
            labels=dict(x="Thème", y="Sentiment", color="Pourcentage"),
            x=sentiment_theme_pct.index,
            y=sentiment_theme_pct.columns,
            color_continuous_scale='RdYlGn',
            title="Répartition des sentiments par thème (%)",
            text_auto='.1f'
        )
        fig_heatmap.update_xaxes(tickangle=45)
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col_sent2:
        # Score de sentiment par thème
        sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        df_exploded['sentiment_score'] = df_exploded['sentiment'].map(sentiment_mapping)
        theme_sentiment_score = df_exploded.groupby('themes')['sentiment_score'].mean().sort_values()
        
        fig_theme_score = px.bar(
            x=theme_sentiment_score.values,
            y=theme_sentiment_score.index,
            orientation='h',
            title="Score moyen de sentiment par thème",
            labels={'x': 'Score (-1 à 1)', 'y': 'Thème'},
            color=theme_sentiment_score.values,
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        fig_theme_score.update_layout(height=400)
        st.plotly_chart(fig_theme_score, use_container_width=True)
    
    # Comparaison thématique entre services
    st.subheader("🔄 Comparaison thématique Orange Money vs WAVE")
    
    if len(services) >= 2:
        col_comp1, col_comp2 = st.columns(2)
        
        for idx, service in enumerate(services[:2]):
            df_service_theme = df_exploded[df_exploded['Service'] == service]
            theme_sentiment_service = df_service_theme.groupby(['themes', 'sentiment']).size().reset_index(name='count')
            
            fig_service_theme = px.bar(
                theme_sentiment_service,
                x='themes',
                y='count',
                color='sentiment',
                title=f"Thèmes et sentiments - {service}",
                labels={'themes': 'Thème', 'count': 'Nombre'},
                color_discrete_map={
                    'POSITIVE': '#2ecc71',
                    'NEUTRAL': '#3498db',
                    'NEGATIVE': '#e74c3c'
                },
                barmode='stack'
            )
            fig_service_theme.update_xaxes(tickangle=45)
            fig_service_theme.update_layout(height=400)
            
            if idx == 0:
                col_comp1.plotly_chart(fig_service_theme, use_container_width=True)
            else:
                col_comp2.plotly_chart(fig_service_theme, use_container_width=True)
    
    # Top commentaires par thème
    st.subheader("📝 Exemples de commentaires par thème")
    
    selected_theme = st.selectbox(
        "Sélectionner un thème pour voir des exemples",
        options=sorted(df_exploded['themes'].unique())
    )
    
    selected_sentiment_example = st.radio(
        "Filtrer par sentiment",
        options=['Tous'] + list(df_filtered['sentiment'].unique()),
        horizontal=True
    )
    
    # Filtrer les commentaires
    df_theme_examples = df_exploded[df_exploded['themes'] == selected_theme].copy()
    
    if selected_sentiment_example != 'Tous':
        df_theme_examples = df_theme_examples[df_theme_examples['sentiment'] == selected_sentiment_example]
    
    # Afficher les exemples
    if len(df_theme_examples) > 0:
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        with col_ex1:
            st.metric("Nombre de commentaires", len(df_theme_examples))
        with col_ex2:
            avg_words = df_theme_examples['nb_mots'].mean()
            st.metric("Moy. mots", f"{avg_words:.1f}")
        with col_ex3:
            pos_rate = (df_theme_examples['sentiment'] == 'POSITIVE').sum() / len(df_theme_examples) * 100
            st.metric("Taux positif", f"{pos_rate:.1f}%")
        
        st.dataframe(
            df_theme_examples[['Service', 'clean_sentence', 'sentiment', 'nb_mots']].head(10),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Aucun commentaire trouvé pour cette combinaison thème/sentiment")
    
    # Matrice de co-occurrence des thèmes
    st.subheader("🔗 Co-occurrence des thèmes")
    
    # Créer une matrice de co-occurrence
    from itertools import combinations
    
    theme_pairs = []
    for themes_list in df_filtered['themes']:
        if len(themes_list) > 1:
            for pair in combinations(sorted(themes_list), 2):
                theme_pairs.append(pair)
    
    if theme_pairs:
        cooccurrence = Counter(theme_pairs)
        cooc_df = pd.DataFrame(cooccurrence.most_common(15), columns=['Paire de thèmes', 'Fréquence'])
        cooc_df['Paire'] = cooc_df['Paire de thèmes'].apply(lambda x: f"{x[0]} + {x[1]}")
        
        fig_cooc = px.bar(
            cooc_df,
            x='Fréquence',
            y='Paire',
            orientation='h',
            title="Top 15 combinaisons de thèmes",
            color='Fréquence',
            color_continuous_scale='purples'
        )
        st.plotly_chart(fig_cooc, use_container_width=True)
    else:
        st.info("Pas assez de commentaires avec plusieurs thèmes pour cette sélection")
    
    # Table des commentaires
    st.markdown("---")
    st.subheader("📝 Détails des Commentaires")
    
    # Options d'affichage
    col9, col10 = st.columns(2)
    with col9:
        nb_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10)
    with col10:
        sort_by = st.selectbox(
            "Trier par",
            ['sentiment', 'nb_mots', 'cluster', 'Service']
        )
    
    # Affichage de la table
    display_columns = ['Service', 'clean_sentence', 'sentiment', 'cluster', 'nb_mots']
    st.dataframe(
        df_filtered[display_columns].sort_values(by=sort_by).head(nb_rows),
        use_container_width=True,
        hide_index=True
    )
    
    # Export des données filtrées
    st.markdown("---")
    st.subheader("💾 Export des Données")
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv,
        file_name='donnees_filtrees.csv',
        mime='text/csv',
    )
    
except FileNotFoundError:
    st.error("❌ Fichier 'final_database.csv' non trouvé. Assurez-vous qu'il est dans le même répertoire que l'application.")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données : {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Dashboard créé avec Streamlit - Analyse de sentiments Orange Money & WAVE*")