import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import numpy as np
import os
import time

# Configuration
st.set_page_config(
    page_title="Détection de Faux et Vrais Billets",
    page_icon="💰",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
.stApp {
    background-color: #fff9e6;
}
.header {
    background-color: #d4a017;
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.billet-container {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    gap: 20px;
}
.billet-image {
    width: 200px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.metric-box {
    background-color: #fff9e6;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #d4a017;
    text-align: center;
    margin: 10px;
}
.feature-details {
    background-color: #fff9e6;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Chargement du modèle et du scaler
try:
    model = joblib.load('random_forest_model.sav')
    scaler = joblib.load('scaler.sav')
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {str(e)}")
    st.stop()

# Chemins des images
GENUINE_IMAGE_PATH = os.path.join("images", "vrai.png")
FAKE_IMAGE_PATH = os.path.join("images", "faux.png")

# Titre
st.markdown("""
<div class="header">
    <h1 style='text-align: center; margin: 0;'>Détection Automatique de Faux Billets</h1>
    <p style='text-align: center; margin: 0;'>Analyse basée sur les caractéristiques géométriques</p>
</div>
""", unsafe_allow_html=True)

# Fonction de prédiction locale
def local_predict(data):
    """Effectue les prédictions directement sans API"""
    required_columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    
    # Prétraitement
    data = data[required_columns].apply(pd.to_numeric, errors='coerce')
    data = data.fillna(data.median())
    
    # Prédiction
    X_scaled = scaler.transform(data)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Formatage des résultats
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            "id": i + 1,
            "prediction": "Genuine" if pred else "Fake",
            "probability": float(prob[1] if pred else prob[0]),
            "features": data.iloc[i].to_dict()
        })
    
    # Calcul des statistiques
    genuine_count = int(sum(predictions))
    fake_count = int(len(predictions) - genuine_count)
    
    return {
        "predictions": results,
        "stats": {
            "total": len(predictions),
            "genuine": genuine_count,
            "fake": fake_count,
            "genuine_percentage": round(genuine_count / len(predictions) * 100, 2),
            "fake_percentage": round(fake_count / len(predictions) * 100, 2)
        }
    }

def display_billet_details(billet):
    """Affiche les détails d'un billet"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            image = Image.open(GENUINE_IMAGE_PATH if billet['prediction'] == 'Genuine' else FAKE_IMAGE_PATH)
            st.image(image, width=200)
        except Exception as e:
            st.error(f"Image non trouvée: {e}")
            st.image("https://via.placeholder.com/200", width=200)
    
    with col2:
        st.subheader(f"Billet #{billet['id']}")
        if billet['prediction'] == 'Genuine':
            st.success(f"Authentique ({(billet['probability']*100):.1f}% de confiance)")
        else:
            st.error(f"Contrefait ({(billet['probability']*100):.1f}% de confiance)")
        
        features = billet.get('features', {})
        st.markdown(f"""
        <div class="feature-details">
            <strong>Caractéristiques:</strong><br>
            - Longueur: {features.get('length', 'N/A'):.2f} mm<br>
            - Hauteur gauche: {features.get('height_left', 'N/A'):.2f} mm<br>
            - Hauteur droite: {features.get('height_right', 'N/A'):.2f} mm<br>
            - Marge supérieure: {features.get('margin_up', 'N/A'):.2f} mm<br>
            - Marge inférieure: {features.get('margin_low', 'N/A'):.2f} mm<br>
            - Diagonale: {features.get('diagonal', 'N/A'):.2f} mm
        </div>
        """, unsafe_allow_html=True)

# Interface principale
st.markdown("---")
st.markdown("""
### À propos de cette application
Cet outil analyse les caractéristiques géométriques des billets en euros pour détecter les contrefaçons.
Les valeurs manquantes sont automatiquement remplacées par la médiane des valeurs existantes.
""")

uploaded_file = st.file_uploader(
    "Importez un fichier CSV contenant les mesures des billets", 
    type=["csv", "txt"],
    help="Le fichier doit contenir 7 colonnes séparées par ';' sans en-tête : is_genuine,diagonal,height_left,height_right,margin_low,margin_up,length"
)

if uploaded_file is not None:
    try:
        # Lecture du CSV avec séparateur ; et sans en-tête
        data = pd.read_csv(uploaded_file, sep=';', header=None)
        
        # Attribution des noms de colonnes
        data.columns = ['is_genuine', 'diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
        
        # Conversion de la colonne is_genuine
        data['is_genuine'] = data['is_genuine'].astype(bool)
        
        st.subheader("Aperçu des données")
        st.dataframe(data.head())
        
        if st.button("Lancer la détection", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Animation de chargement
                for percent_complete in range(0, 101, 10):
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Analyse en cours... {percent_complete}%")
                    time.sleep(0.1)
                
                # Prédiction locale (on exclut la colonne is_genuine)
                results = local_predict(data.drop(columns=['is_genuine']))
                
                progress_bar.empty()
                status_text.empty()
                
                # Affichage des résultats
                st.subheader("Résultats globaux")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Total</h3>
                        <p style="font-size: 24px; font-weight: bold;">{results['stats']['total']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Vrais</h3>
                        <p style="font-size: 24px; font-weight: bold;">{results['stats']['genuine']} ({results['stats']['genuine_percentage']:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>Faux</h3>
                        <p style="font-size: 24px; font-weight: bold;">{results['stats']['fake']} ({results['stats']['fake_percentage']:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Graphique
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#d4a017', '#f5d76e']
                ax.pie(
                    [results['stats']['genuine'], results['stats']['fake']],
                    labels=['Vrais', 'Faux'],
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 12}
                )
                ax.axis('equal')
                st.pyplot(fig)
                
                # Détails par billet
                st.subheader("Analyse détaillée par billet")
                for billet in results['predictions']:
                    display_billet_details(billet)
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Erreur lors de l'analyse: {str(e)}")
    
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier: {str(e)}")