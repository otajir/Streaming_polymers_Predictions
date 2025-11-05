"""
Script utilitaire pour sauvegarder les features utilisées lors de l'entraînement
À exécuter APRÈS l'entraînement de chaque modèle
"""
import joblib
import json

def save_model_features(model, model_name, output_path="model_features.json"):
    """
    Sauvegarde les features attendues par un modèle
    
    Args:
        model: Le modèle entraîné (LightGBM, XGBoost, etc.)
        model_name: Nom du modèle (ex: "MELT_ANA", "IZOD_ANA")
        output_path: Chemin du fichier JSON de sortie
    """
    # Détecter les features selon le type de modèle
    if hasattr(model, 'feature_name_'):
        features = model.feature_name_
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
        features = model.get_booster().feature_names
    elif hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
    else:
        raise ValueError("Impossible de détecter les features du modèle")
    
    # Charger le fichier existant ou créer un nouveau dict
    try:
        with open(output_path, 'r') as f:
            all_features = json.load(f)
    except FileNotFoundError:
        all_features = {}
    
    # Ajouter les features de ce modèle
    all_features[model_name] = features
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(all_features, f, indent=2)
    
    print(f"✅ Features de {model_name} sauvegardées: {len(features)} colonnes")
    return features


# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================
if __name__ == "__main__":
    # Après avoir entraîné vos modèles, faites ceci:
    
    # 1. Charger les modèles
    model_melt = joblib.load("LightGBM_MELT_ANA_model.pkl")
    model_izod = joblib.load("BestModel_IZOD_ANA_XGBoost.pkl")
    model_flex = joblib.load("BestModel_FLEX_ANA_LightGBM_v2.pkl")
    
    # 2. Sauvegarder les features
    save_model_features(model_melt, "MELT_ANA")
    save_model_features(model_izod, "IZOD_ANA")
    save_model_features(model_flex, "FLEX_ANA")
    
    print("\n📄 Fichier 'model_features.json' créé avec succès!")