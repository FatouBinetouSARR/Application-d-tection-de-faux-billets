# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import base64
import io
import os
import numpy as np
from time import time

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
if 'last_update' not in st.session_state:
    st.session_state.last_update = 0

# Chemins des images locales
GENUINE_IMG_PATH = os.path.join("images", "vrai.png")
FAKE_IMG_PATH = os.path.join("images", "faux.png")

# Fonction pour charger les images
@st.cache_resource
def load_images():
    try:
        with open(GENUINE_IMG_PATH, "rb") as img_file:
            genuine_img = base64.b64encode(img_file.read()).decode('utf-8')
        with open(FAKE_IMG_PATH, "rb") as img_file:
            fake_img = base64.b64encode(img_file.read()).decode('utf-8')
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
    --secondary: #fff9e6;
    --success: #28a745;
    --danger: #dc3545;
    font-family: 'Arial', sans-serif;
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
.genuine-card { border-left: 4px solid var(--success); }
.fake-card { border-left: 4px solid var(--danger); }
.stat-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.stat-card {
    text-align: center;
    padding: 0.8rem;
    border-radius: 8px;
    background-color: white;
    border: 1px solid var(--primary);
}
.probability-bar {
    height: 8px;
    border-radius: 4px;
    background: #e9ecef;
    margin: 0.3rem 0;
}
.billet-image {
    border-radius: 6px;
    width: 100px;
    border: 2px solid white;
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application
st.markdown("""
<div class="header">
    <h1 style='text-align: center; margin: 0;'>Détection Automatique de Faux Billets</h1>
    <p style='text-align: center; margin: 0;'>Analyse basée sur les caractéristiques géométriques</p>
</div>
""", unsafe_allow_html=True)

# Interface principale
st.markdown("---")
st.markdown("### À propos")
st.markdown("""
Bienvenue dans notre application de détection automatique de faux billets en euros.

Cette application permet d'analyser les caractéristiques géométriques des billets pour déterminer leur authenticité avec une précision de 98%.

**Fonctionnalités :**
- Analyse de 6 paramètres géométriques
- Interface intuitive
- Résultats détaillés avec niveaux de confiance
""")

# Section Analyse
uploaded_file = st.file_uploader(
    "📤 Faites glisser et déposez votre fichier CSV ici", 
    type=["csv"],
    help="Le fichier doit contenir les colonnes: length, height_left, height_right, margin_up, margin_low, diagonal"
)

if uploaded_file is not None:
    try:
        @st.cache_data(ttl=3600, show_spinner="Chargement des données...")
        def load_data(uploaded_file):
            return pd.read_csv(uploaded_file, sep=';')
        
        st.session_state.df = load_data(uploaded_file)
        
        with st.expander("🔍 Aperçu des données (cliquez pour développer)", expanded=False):
            st.dataframe(st.session_state.df.head(), height=210, use_container_width=True)
        
        if st.button("🔎 Lancer l'analyse", key="analyze_btn", type="primary"):
            with st.spinner("Analyse en cours... Veuillez patienter"):
                try:
                    file_bytes = io.BytesIO(uploaded_file.getvalue())
                    files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                    
                    response = requests.post(
                        "http://localhost:8000/predict", 
                        files=files, 
                        timeout=10
                    )
                    response.raise_for_status()
                    
                    st.session_state.results = response.json()
                    st.session_state.last_update = time()
                    st.toast("✅ Analyse terminée avec succès !")
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Erreur de connexion à l'API: {str(e)}")
                    st.session_state.results = None
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                    st.session_state.results = None

    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier: {str(e)}")

# Affichage des résultats
if st.session_state.results:
    predictions = st.session_state.results.get('predictions', [])
    
    if not predictions:
        st.warning("⚠️ Aucun résultat à afficher")
    else:
        st.markdown("## 📊 Résultats de la détection")
        st.markdown("---")
        
        # Affichage paginé des billets
        page_size = 10
        page_number = st.number_input('Page', min_value=1, max_value=int(np.ceil(len(predictions)/page_size)), value=1)
        start_idx = (page_number-1)*page_size
        end_idx = start_idx + page_size
        
        for pred in predictions[start_idx:end_idx]:
            is_genuine = pred.get('prediction', '').lower() == 'genuine'
            prob = pred.get('probability', 0)
            prob = prob if is_genuine else (1 - prob)
            prob_percent = min(100, max(0, prob * 100))  # Garantit entre 0 et 100
            color = "#28a745" if is_genuine else "#dc3545"
            status = "Authentique ✅" if is_genuine else "Faux ❌"
            
            st.markdown(f"""
            <div class="card {'genuine-card' if is_genuine else 'fake-card'}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0; color: {color}; font-size: 1.1rem;">Billet n°{pred.get('id', 'N/A')}</h3>
                        <p style="margin: 0.3rem 0; font-size: 0.9rem;">
                            Statut: <strong>{status}</strong>
                        </p>
                        <p style="margin: 0.3rem 0; font-size: 0.9rem;">
                            Confiance: <strong>{prob_percent:.1f}%</strong>
                        </p>
                        <div class="probability-bar">
                            <div style="height: 100%; width: {prob_percent}%; background: {color}; border-radius: 4px;"></div>
                        </div>
                    </div>
                    <img src="data:image/png;base64,{genuine_img if is_genuine else fake_img}" 
                         class="billet-image">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques
        st.markdown("---")
        st.markdown("### 📈 Statistiques globales")
        
        genuine_count = sum(1 for p in predictions if p.get('prediction', '').lower() == 'genuine')
        fake_count = len(predictions) - genuine_count
        
        st.markdown("""
        <div class="stat-container">
            <div class="stat-card">
                <h3>Total</h3>
                <p style="font-size: 24px; color: var(--primary);">{len(predictions)}</p>
            </div>
            <div class="stat-card">
                <h3>Authentiques</h3>
                <p style="font-size: 24px; color: var(--success);">{genuine_count} ({genuine_count/len(predictions)*100:.1f}%)</p>
            </div>
            <div class="stat-card">
                <h3>Faux</h3>
                <p style="font-size: 24px; color: var(--danger);">{fake_count} ({fake_count/len(predictions)*100:.1f}%)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphiques
        st.markdown("### 📊 Visualisations")
        
        tab1, tab2 = st.tabs(["Répartition", "Confiance moyenne"])
        
        with tab1:
            fig_pie = px.pie(
                names=['Authentiques', 'Faux'],
                values=[genuine_count, fake_count],
                color_discrete_sequence=["#28a745", "#dc3545"],
                hole=0.3,
                template="plotly_white"
            )
            fig_pie.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            genuine_probs = [p.get('probability', 0) for p in predictions if p.get('prediction', '').lower() == 'genuine']
            fake_probs = [1-p.get('probability', 0) for p in predictions if p.get('prediction', '').lower() == 'fake']
            
            avg_genuine = np.mean(genuine_probs)*100 if genuine_probs else 0
            avg_fake = np.mean(fake_probs)*100 if fake_probs else 0
            
            fig_bar = px.bar(
                x=['Authentiques', 'Faux'],
                y=[avg_genuine, avg_fake],
                color=['Authentiques', 'Faux'],
                color_discrete_map={'Authentiques': '#28a745', 'Faux': '#dc3545'},
                text=[f"{avg_genuine:.1f}%", f"{avg_fake:.1f}%"],
                labels={'x': '', 'y': 'Confiance moyenne (%)'},
                template="plotly_white"
            )
            fig_bar.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                height=300,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_bar, use_container_width=True)