import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings
import urllib.parse
from pymongo import MongoClient
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# -----------------------------------------------------------
# 🗄️ Configuration MongoDB - NOUVELLE BASE DE DONNÉES
# -----------------------------------------------------------
@st.cache_resource
def init_mongodb():
    """
    Initialise la connexion MongoDB pour le projet PolySmart
    
    Configuration à personnaliser :
    1. Remplacez les credentials par les vôtres
    2. Changez le nom de la base de données si nécessaire
    3. Changez le nom de la collection si nécessaire
    """
    try:
        # ⚠️ REMPLACEZ CES VALEURS PAR VOS PROPRES CREDENTIALS
        mongo_username = urllib.parse.quote_plus("exxelUser")  # ← Changez ici
        mongo_password = urllib.parse.quote_plus("123abcA@")  # ← Changez ici
        mongo_cluster = "exxel.npoktth.mongodb.net"                 # ← Changez ici (ex: cluster0.abc123.mongodb.net)
        
        # URL de connexion MongoDB
        mongo_url = f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_cluster}/?retryWrites=true&w=majority"
        
        # Connexion au client MongoDB
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # 📊 NOUVELLE BASE DE DONNÉES pour ce projet
        db = client['polysmart_database']  # ← Changez le nom de la base ici
        
        # 📦 NOUVELLE COLLECTION pour stocker les prédictions
        collection = db['polymer_predictions']  # ← Changez le nom de la collection ici
        
        # Test de connexion
        client.server_info()
        
        st.sidebar.success("✅ MongoDB connecté")
        st.sidebar.info(f"📊 Base: `{db.name}`")
        st.sidebar.info(f"📦 Collection: `{collection.name}`")
        
        return collection
        
    except Exception as e:
        st.sidebar.error(f"❌ Erreur MongoDB : {e}")
        st.sidebar.warning("⚠️ Les prédictions ne seront pas sauvegardées")
        return None

# -----------------------------------------------------------
# ⚙️ Initialisation
# -----------------------------------------------------------
st.set_page_config(page_title="Polymers Prediction", layout="wide", page_icon="🧪")
st.title("Prédiction des propriétés des mélanges polymères")
st.markdown("Plateforme de prédiction **MELT / IZOD / FLEX** avec sauvegarde automatique")

# -----------------------------------------------------------
# 📦 Chargement des modèles et features
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "MELT_ANA": joblib.load("LightGBM_MELT_ANA_model.pkl"),
        "IZOD_ANA": joblib.load("BestModel_IZOD_ANA_XGBoost.pkl"),
        "FLEX_ANA": joblib.load("BestModel_FLEX_ANA_LightGBM_v2.pkl")
    }
    return models

@st.cache_data
def load_expected_features():
    """Charge les features attendues depuis le fichier JSON"""
    try:
        with open("model_features.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Fichier 'model_features.json' introuvable!")
        return None

models = load_models()
expected_features_dict = load_expected_features()
mongodb_collection = init_mongodb()

# -----------------------------------------------------------
# 📊 Sidebar - Statistiques MongoDB
# -----------------------------------------------------------
if mongodb_collection is not None:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Statistiques")
        try:
            total_predictions = mongodb_collection.count_documents({})
            st.metric("Total prédictions", total_predictions)
            
            if total_predictions > 0:
                last_pred = list(mongodb_collection.find().sort("timestamp", -1).limit(1))[0]
                last_time = datetime.fromisoformat(last_pred["timestamp"])
                st.metric("Dernière prédiction", last_time.strftime("%d/%m %H:%M"))
        except:
            pass

# -----------------------------------------------------------
# 1️⃣ Sélection du nombre d'items
# -----------------------------------------------------------
st.header("1️⃣ Définir la composition du mélange")
nb_items = st.selectbox("Nombre d'items dans la recette :", [2, 3])

# Liste complète incluant tous les items possibles
ITEMS_LIST = [
    "PP", "PPH", "PPG", "HDPE", "LDPE", "LLDPE", "MDPE",
    "PE/PP", "HIPS", "PS", "ABS", "PC", "TPO", "EVA",
    "CONPP", "CONPS", "CONPE", "CONABS", "MIPS", "PEROX",
    "ADDITIFS", "HMW"
]

# -----------------------------------------------------------
# 🧩 2️⃣ Fiches de saisie dynamiques
# -----------------------------------------------------------
recette_detaillee = []

for i in range(nb_items):
    st.subheader(f"🧬 Item {i+1}")
    col1, col2, col3 = st.columns(3)
    with col1:
        item = st.selectbox(f"Type d'item #{i+1}", ITEMS_LIST, key=f"item_{i}")
        pourc = st.number_input(f"Fraction massique de {item}", 0.0, 1.0, 0.0, 0.01, key=f"frac_{i}")
        lignep = st.selectbox(f"Ligne de production", ["Melange", "Extrusion"], key=f"lignep_{i}")
        couleur = st.selectbox("Couleur", ["BK", "CL", "GREY", "MIX", "NAT", "WH"], key=f"coul_{i}")
    with col2:
        i_f = st.selectbox("Indice de forme (I_F)", ["B", "P", "PR", "PW", "RG", "RP"], key=f"if_{i}")
        i_cm = st.selectbox("I_CM (Procédé de moulage)", ["EXT", "HM", "INJ", "OFF", "X"], key=f"icm_{i}")
        i_g = st.selectbox("I_G (Gamme de procédé)", ["1", "A", "B", "C", "X", "Z"], key=f"ig_{i}")
        melt = st.number_input("MELT_ANA", 0.0, 100.0, 0.0, 0.1, key=f"melt_{i}")
    with col3:
        dens = st.number_input("DENS_ANA", 0.0, 2.0, 0.0, 0.01, key=f"dens_{i}")
        cend = st.number_input("CEND_ANA", 0.0, 10.0, 0.0, 0.1, key=f"cend_{i}")
        izod = st.number_input("IZOD_ANA", 0.0, 50.0, 0.0, 0.1, key=f"izod_{i}")
        flex = st.number_input("FLEX_ANA", 0.0, 500000.0, 0.0, 1000.0, key=f"flex_{i}")
        tract = st.number_input("TRACT_ANA", 0.0, 500000.0, 0.0, 1000.0, key=f"tract_{i}")

    recette_detaillee.append({
        "Items": item, "%_additifs": pourc,
        "lignep": lignep, "I_F": i_f, "COULEUR": couleur,
        "I_CM": i_cm, "I_G": i_g,
        "MELT_ANA": melt, "DENS_ANA": dens, "CEND_ANA": cend,
        "IZOD_ANA": izod, "FLEX_ANA": flex, "TRACT_ANA": tract
    })

# -----------------------------------------------------------
# 3️⃣ Traçabilité
# -----------------------------------------------------------
st.header("3️⃣ Traçabilité")
colA, colB, colC = st.columns(3)
with colA:
    wo_no = st.text_input("WO-NO (numéro de lot / production)", placeholder="WO-2025-001")
with colB:
    valeur_reelle = st.number_input("Valeur réelle mesurée (facultatif)", 0.0, 1_000_000.0, 0.0, 0.1)

# -----------------------------------------------------------
# 4️⃣ Calcul de la recette agrégée
# -----------------------------------------------------------
if st.button("⚙️ Calculer et lancer la prédiction", type="primary"):

    df_items = pd.DataFrame(recette_detaillee)
    st.subheader("📋 Données brutes saisies")
    st.dataframe(df_items, use_container_width=True)

    # --- Agrégation pondérée ---
    num_cols = ["MELT_ANA", "DENS_ANA", "CEND_ANA", "IZOD_ANA", "FLEX_ANA", "TRACT_ANA"]
    recette_agregee = pd.DataFrame({
        c + "_conso": [np.nansum(df_items[c] * df_items["%_additifs"]) / df_items["%_additifs"].sum()]
        for c in num_cols
    })

    # --- Variables dominantes ---
    main_row = df_items.loc[df_items["%_additifs"].idxmax()]
    recette_agregee["lignep_conso"] = main_row["lignep"]
    recette_agregee["I_F_conso"] = main_row["I_F"]
    recette_agregee["COULEUR_conso"] = main_row["COULEUR"]
    recette_agregee["I_CM_conso"] = main_row["I_CM"]
    recette_agregee["I_G_conso"] = main_row["I_G"]

    # --- Indicateurs composition : TOUS les items possibles ---
    all_possible_items = [
        "PP", "PPH", "PPG", "HDPE", "LDPE", "LLDPE", "MDPE",
        "PE/PP", "HIPS", "PS", "ABS", "PC", "TPO", "EVA",
        "CONPP", "CONPS", "CONPE", "CONABS", "MIPS", "PEROX",
        "ADDITIFS", "HMW"
    ]
    
    for item in all_possible_items:
        # Normaliser le nom pour PE/PP -> PE_PP
        item_name = item.replace("/", "_")
        recette_agregee[f"item_{item_name}"] = df_items.loc[
            df_items["Items"] == item, "%_additifs"
        ].sum()

    # --- Indicateurs PEROX ---
    recette_agregee["has_perox"] = 1 if "PEROX" in df_items["Items"].values else 0
    recette_agregee["frac_perox"] = df_items.loc[df_items["Items"] == "PEROX", "%_additifs"].sum()

    # --- Ratios physiques COMPLETS ---
    eps = 1e-6
    
    recette_agregee["ratio_flex_tract"] = recette_agregee["FLEX_ANA_conso"] / (recette_agregee["TRACT_ANA_conso"] + eps)
    recette_agregee["ratio_dens_melt"] = recette_agregee["DENS_ANA_conso"] / (recette_agregee["MELT_ANA_conso"] + eps)
    recette_agregee["ratio_izod_melt"] = recette_agregee["IZOD_ANA_conso"] / (recette_agregee["MELT_ANA_conso"] + eps)
    recette_agregee["ratio_izod_flex"] = recette_agregee["IZOD_ANA_conso"] / (recette_agregee["FLEX_ANA_conso"] + eps)
    recette_agregee["ratio_flex_cend"] = recette_agregee["FLEX_ANA_conso"] / (recette_agregee["CEND_ANA_conso"] + eps)
    recette_agregee["ratio_melt_flex"] = recette_agregee["MELT_ANA_conso"] / (recette_agregee["FLEX_ANA_conso"] + eps)
    recette_agregee["ratio_dens_flex"] = recette_agregee["DENS_ANA_conso"] / (recette_agregee["FLEX_ANA_conso"] + eps)

    # Sauvegarder une copie avant encodage pour MongoDB
    recette_agregee_original = recette_agregee.copy()

    # -------------------------------------------------------
    # 🧩 Encodage des variables catégorielles
    # -------------------------------------------------------
    try:
        from sklearn.preprocessing import LabelEncoder
        cat_cols = ["lignep_conso", "I_F_conso", "COULEUR_conso", "I_CM_conso", "I_G_conso"]

        for col in cat_cols:
            if col in recette_agregee.columns:
                recette_agregee[col] = recette_agregee[col].astype(str)
                le = LabelEncoder()
                recette_agregee[col] = le.fit_transform(recette_agregee[col])

        recette_agregee = recette_agregee.apply(pd.to_numeric, errors='coerce').fillna(0)

    except Exception as e:
        st.error(f"⚠️ Erreur d'encodage : {e}")

    # -------------------------------------------------------
    # 🔮 Prédictions
    # -------------------------------------------------------
    predictions = {}
    
    with st.expander("🔍 Détails techniques des prédictions", expanded=False):
        for target, model in models.items():
            try:
                if expected_features_dict and target in expected_features_dict:
                    expected_features = expected_features_dict[target]
                    st.write(f"**{target}**: {len(expected_features)} features requises")
                else:
                    st.error(f"❌ Features manquantes pour {target}")
                    continue
                
                current_features = set(recette_agregee.columns)
                expected_features_set = set(expected_features)
                
                missing = expected_features_set - current_features
                
                if missing:
                    st.warning(f"⚠️ {target}: Ajout de {len(missing)} features manquantes")
                    for feat in missing:
                        recette_agregee[feat] = 0
                
                X_pred = recette_agregee[expected_features]
                pred = model.predict(X_pred)[0]
                predictions[target] = round(float(pred), 3)
                st.success(f"✅ {target}: {predictions[target]}")
                
            except Exception as e:
                st.error(f"❌ {target}: {str(e)}")
                predictions[target] = "N/A"

    # -------------------------------------------------------
    # Affichage des résultats
    # -------------------------------------------------------
    st.subheader("📈 Résultats de prédiction")
    col1, col2, col3 = st.columns(3)
    
    melt_val = predictions.get("MELT_ANA", "N/A")
    izod_val = predictions.get("IZOD_ANA", "N/A")
    flex_val = predictions.get("FLEX_ANA", "N/A")
    
    col1.metric("MELT_ANA", f"{melt_val} g/10min", delta=None if valeur_reelle == 0 else f"{melt_val - valeur_reelle:.2f}")
    col2.metric("IZOD_ANA", f"{izod_val} kJ/m²")
    col3.metric("FLEX_ANA", f"{flex_val} MPa")

    # -------------------------------------------------------
    # 💾 SAUVEGARDE DANS MONGODB - NOUVELLE BASE
    # -------------------------------------------------------
    if mongodb_collection is not None:
        try:
            # 📦 Document complet à sauvegarder
            document = {
                # Traçabilité
                "wo_no": wo_no if wo_no else f"AUTO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                
                # Configuration
                "nb_items": nb_items,
                
                # Données détaillées de CHAQUE item
                "items_detail": df_items.to_dict('records'),
                
                # Recette agrégée (valeurs originales avant encodage)
                "recette_agregee": recette_agregee_original.to_dict('records')[0],
                
                # Résultats des prédictions
                "predictions": {
                    "MELT_ANA": melt_val if melt_val != "N/A" else None,
                    "IZOD_ANA": izod_val if izod_val != "N/A" else None,
                    "FLEX_ANA": flex_val if flex_val != "N/A" else None
                },
                
                # Valeur réelle (traçabilité)
                "valeur_reelle": valeur_reelle if valeur_reelle > 0 else None,
                
                # Métadonnées
                "success": all(v != "N/A" for v in [melt_val, izod_val, flex_val]),
                "version": "1.0",
                "app_name": "PolySmart AI"
            }
            
            # Insertion dans MongoDB
            result = mongodb_collection.insert_one(document)
            
            st.success(f"💾 ✅ Données sauvegardées dans MongoDB")
            st.info(f"📝 Document ID: `{result.inserted_id}`")
            
        except Exception as e:
            st.error(f"❌ Erreur MongoDB : {e}")
            st.warning("⚠️ Les données n'ont pas été sauvegardées")
    else:
        st.warning("⚠️ MongoDB non connecté - Les données ne sont pas sauvegardées")

    # -------------------------------------------------------
    # ✅ Message final
    # -------------------------------------------------------
    if all(v != "N/A" for v in [melt_val, izod_val, flex_val]):
        st.success(f"✅ Prédiction réalisée avec succès pour : **{wo_no if wo_no else 'Lot auto-généré'}**")
    else:
        st.error("⚠️ Certaines prédictions ont échoué")
    
    if valeur_reelle:
        erreur = abs(melt_val - valeur_reelle) if melt_val != "N/A" else None
        if erreur:
            st.info(f"📏 Valeur réelle : **{valeur_reelle}** | Erreur : **{erreur:.3f}**")