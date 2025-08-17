# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
import numpy as np
from time import time
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation des variables de session
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Chargement du modèle et du scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.sav')
        scaler = joblib.load('scaler.sav')
        return model, scaler
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {str(e)}")
        return None, None

model, scaler = load_model()

# Chemins des images locales
GENUINE_IMG_PATH = os.path.join("images", "vrai.png")
FAKE_IMG_PATH = os.path.join("images", "faux.png")

# Fonction pour charger les images
@st.cache_resource
def load_images():
    try:
        # Charger les images une fois et les stocker en mémoire
        genuine_img = open(GENUINE_IMG_PATH, "rb").read()
        fake_img = open(FAKE_IMG_PATH, "rb").read()
        return genuine_img, fake_img
    except Exception as e:
        st.error(f"Erreur de chargement des images: {str(e)}")
        return None, None

genuine_img, fake_img = load_images()

# CSS optimisé
st.markdown("""
            
<style>
:root {
    --primary: #d4a017;
    --primary-dark: #b38a14;
    --secondary: #fff9e6;
    --genuine-color: #a37d12;
    --fake-color: #5a3921;
    
}

.genuine-text { color: var(--genuine-color) !important; }
.fake-text { color: var(--fake-color) !important; }

button {
    background-color: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

button:hover {
    background-color: var(--primary-dark) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

.stButton>button {
    width: 100%;
}

.stApp { background-color: var(--secondary); }

.header {
    background-color: var(--primary);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.card {
    background-color: white;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.genuine-card { 
    border-left: 4px solid var(--genuine-color);
    background-color: rgba(163, 125, 18, 0.05);
}

.fake-card { 
    border-left: 4px solid var(--fake-color);
    background-color: rgba(90, 57, 33, 0.05);
}


.billet-card {
        display: flex;
        padding: 20px;
        margin-bottom: 20px;
        background: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        height: 220px; 
        overflow: hidden;
    }

    .billet-info {
        flex: 1;
        padding-right: 15px;
        display: flex;
        flex-direction: column;
        justify-content: space-between; 
    }

.billet-image-container {
    flex-shrink: 0;
    width: 160px;
    height: 160px; 
    display: flex;
    align-items: center;
    justify-content: center;
    }

.billet-image {
    max-width: 100%;
    max-height: 100%;
    border-radius: 8px;
    border: 3px solid #eee;
}

.probability-bar {
    height: 6px;
    background: #f0f0f0;
    border-radius: 3px;
    margin: 10px 0;
    overflow: hidden;
}

.probability-fill.genuine {
}
.probability-fill.fake {
    background: var(--fake-color) 
}

""", unsafe_allow_html=True)

# Titre de l'application
st.markdown("""
<div class="header">
    <h1 style='text-align: center; margin: 0;'>Détection Automatique de Faux Billets</h1>
    <p style='text-align: center; margin: 0;'>Analyse basée sur les caractéristiques géométriques</p>
</div>
""", unsafe_allow_html=True)

# Section à propos
st.markdown("---")
st.markdown("### À propos")
st.markdown("""
Contexte
Ce projet vise à développer un algorithme de détection automatique de faux billets en euros en utilisant des techniques de machine learning. Il s'inscrit dans une démarche de lutte contre la contrefaçon en exploitant des caractéristiques géométriques mesurables des billets, imperceptibles à l'œil nu mais détectables par une machine.

Objectifs
L'objectif principal est de construire un modèle capable de prédire avec précision si un billet est authentique ou contrefait, en se basant sur six dimensions géométriques :

Longueur (length)

Hauteur à gauche (height_left) et à droite (height_right)

Marges supérieure (margin_up) et inférieure (margin_low)

Diagonale (diagonal).
""")

# Section Analyse
uploaded_file = st.file_uploader(
    "📂 Importez votre fichier CSV", 
    type=["csv"],
    help="Le fichier doit contenir les colonnes: length, height_left, height_right, margin_up, margin_low, diagonal"
)

def predict_data(df):
    """Fonction pour faire les prédictions directement dans Streamlit"""
    try:
        # Nettoyage et vérification des colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        column_mapping = {
            'diagonal': ['diagonal', 'dingmail', 'diagonale'],
            'height_left': ['height_left', 'hauteur_gauche'],
            'height_right': ['height_right', 'hauteur_droite'],
            'margin_low': ['margin_low', 'margin_bow', 'marge_basse'],
            'margin_up': ['margin_up', 'marge_haute'],
            'length': ['length', 'longueur']
        }
        
        for standard_name, variants in column_mapping.items():
            for variant in variants:
                if variant in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                    break
        
        required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes: {missing_cols}")
            return None
        
        # Prétraitement et prédiction
        df = df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(df.median())
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Formatage des résultats
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "id": i + 1,
                "prediction": "Genuine" if pred else "Fake",
                "probability": float(prob[1] if pred else prob[0]),
                "features": df.iloc[i].to_dict()
            })
        
        # Section statistiques: 
        genuine_count = int(sum(predictions))
        return {
            "predictions": results,
            "stats": {
                "total": len(predictions),
                "genuine": genuine_count,
                "fake": len(predictions) - genuine_count,
                "genuine_percentage": round(genuine_count / len(predictions) * 100, 1),
                "fake_percentage": round((len(predictions) - genuine_count) / len(predictions) * 100, 1)
            }
        }
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

if uploaded_file is not None:
    try:
        @st.cache_data(ttl=3600, show_spinner="Chargement des données...")
        def load_data(uploaded_file):
            return pd.read_csv(uploaded_file, sep=';')
        
        st.session_state.df = load_data(uploaded_file)
        
        with st.expander("📄 Aperçu des données (cliquez pour développer)", expanded=False):
            st.dataframe(st.session_state.df.head(), height=210, use_container_width=True)
        
        if st.button("🔎 Analyser", key="analyze_btn", type="primary"):
            with st.spinner("Analyse en cours... "):
                try:
                    if model is None or scaler is None:
                        st.error("Modèle non chargé")
                    else: 
                        st.session_state.results = predict_data(st.session_state.df.copy())
                        st.session_state.last_update = time()
                        st.toast("✅ Analyse terminée avec succès !")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                    st.session_state.results = None
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier: {str(e)}")

# Affichage des résultats
if st.session_state.results:
    predictions = st.session_state.results.get('predictions', [])
    stats = st.session_state.results.get('stats', {})
    
    if not predictions:
        st.warning("⚠️ Aucun résultat à afficher")
    else:
        st.markdown("## 📊 Résultats de la détection")
    
        # Création de 3 colonnes
        col1, col2, col3 = st.columns(3)
        
        # Carte 1 - Le nombre total de billet
        with col1:
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <h3 style="color: #4a6fa5; margin-bottom: 0.5rem;">Total analysés</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #4a6fa5; margin: 0;">{stats.get('total', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Carte 2 - Les vrais billets
        with col2:
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border-left: 5px solid var(--genuine-color);">
                <h3 class="genuine-text" style="margin-bottom: 0.5rem;">Authentiques</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">
                    <span class="genuine-text">{stats.get('genuine', 0)}</span> 
                    <span style="font-size: 1rem;" class="genuine-text">({stats.get('genuine_percentage', 0)}%)</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Carte 3 - Les faux billets
        with col3:
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border-left: 5px solid var(--fake-color);
            ">
                <h3 class="fake-text" style="margin-bottom: 0.5rem;">Faux billets</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0;">
                    <span class="fake-text">{stats.get('fake', 0)}</span> 
                    <span style="font-size: 1rem;" class="fake-text">({stats.get('fake_percentage', 0)}%)</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Section visualisation
        st.markdown("### 📊 Visualisations")
        tab1, tab2 = st.tabs(["📊 Répartition", "📈 Confiance moyenne"])
        
        
        # Diagramme circulaire
        with tab1:  
            fig_pie = px.pie(
                names=['Authentiques', 'Faux'],
                values=[stats.get('genuine', 0), stats.get('fake', 0)],
                color=['Authentiques', 'Faux'],  
                color_discrete_map={
                    'Authentiques': '#a37d12', 
                    'Faux': '#5a3921'          
                }
            )
            fig_pie.update_traces(
                marker=dict(colors=['#a37d12', '#5a3921']),  
                textinfo='percent+label',
                textfont_size=14
            )
            fig_pie.update_layout(
                plot_bgcolor='#fff9e6',
                paper_bgcolor='#fff9e6',
                margin=dict(t=20, b=20, l=20, r=20),
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            try:
                # Calcul des moyennes
                genuine_probs = [p['probability'] for p in predictions if p['prediction'].lower() == 'genuine']
                fake_probs = [p['probability'] for p in predictions if p['prediction'].lower() == 'fake']
                
                avg_genuine = np.mean(genuine_probs)*100 if genuine_probs else 0
                avg_fake = np.mean(fake_probs)*100 if fake_probs else 0
        
                # Création du diagramme
                fig = px.bar(
                    x=['Authentiques', 'Faux'],
                    y=[avg_genuine, avg_fake],
                    color=['Authentiques', 'Faux'],
                    color_discrete_map={'Authentiques': '#a37d12', 'Faux': '#5a3921'},
                    text=[f"{avg_genuine:.1f}%", f"{avg_fake:.1f}%"],
                    labels={'y': 'Confiance moyenne (%)', 'x': ''},
                    height=400
                )
                
                fig.update_layout(
                    yaxis_range=[0, 100],
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_showgrid=False,
                    yaxis_showgrid=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
            except Exception as e:
                st.error(f"Erreur lors de la création du diagramme : {str(e)}")
                st.write("Données utilisées:", predictions)

        st.markdown("---")
        st.markdown("### 🖼️ Visualisation des billets")
    
        predictions_to_display = predictions
        
        
        # Affichons 4 images par ligne
        items_per_row = 4
        num_rows = -(-len(predictions_to_display) // items_per_row) 
        
        for row in range(num_rows):
            cols = st.columns(items_per_row)
            start_idx = row * items_per_row
            
            for col_idx, col in enumerate(cols):
                idx = start_idx + col_idx
                if idx >= len(predictions_to_display):
                    break
                
                pred = predictions_to_display[idx]
                is_genuine = pred.get('prediction', '').lower() == 'genuine'
                prob = pred.get('probability', 0)
                prob_percent = min(100, max(0, prob * 100))
                color = "#a37d12" if is_genuine else "#5a3921" 
                status = "Authentique ✅" if is_genuine else "Faux ❌"
                
                with col:
                    with st.container():
                        st.markdown(f"""
                        <div class="billet-card {'genuine-card' if is_genuine else 'fake-card'}">
                            <div class="billet-info">
                                <div> <!-- Nouveau div pour regrouper le texte -->
                                    <h3 style="margin:0 0 10px 0; color:{color}; font-size:1.2rem;">Billet n°{pred.get('id', 'N/A')}</h3>
                                    <p style="margin:0 0 8px 0; font-size:1.1rem;">Statut: <strong>{status}</strong></p>
                                </div>
                                <div> <!-- Nouveau div pour la partie inférieure -->
                                    <p style="margin:0 0 10px 0; font-size:1.1rem;">Confiance: <strong>{prob_percent:.1f}%</strong></p>
                                    <div class="probability-bar">
                                        <div class="probability-fill" style="width:{prob_percent}%; background:{color};"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="billet-image-container">
                                <img class="billet-image" src="data:image/png;base64,{base64.b64encode(genuine_img if is_genuine else fake_img).decode('utf-8')}">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Affichons  les caractéristiques sous forme de tableau
        st.markdown("---")
        st.markdown("### 🧮 Aperçu des caractéristiques de quelques billets")
        
        features_list = []
        for pred in predictions[:10]:
            features = pred['features'].copy()
            features['id'] = pred['id']
            features['prediction'] = pred['prediction']
            features['probability'] = pred['probability']
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        cols = ['id', 'prediction', 'probability'] + [c for c in df_features.columns if c not in ['id', 'prediction', 'probability']]
        df_features = df_features[cols]
        
        
        st.dataframe(
            df_features.style
            .format("{:.2f}", subset=df_features.select_dtypes(include=['float64']).columns)
            .applymap(lambda x: 'color: #a37d12' if x == 'Genuine' else 'color: #5a3921',
                      subset=['prediction'])
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([{
                'selector': 'thead th',
                'props': [('background-color', '#d4a017'), ('color', 'white')]
            }]),
            height=250,
            use_container_width=True
        )