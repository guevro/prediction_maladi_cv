import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import time

# --- Configuration de la page Streamlit ---
st.set_page_config(
    layout="wide",
    page_title="Pipeline ML : Détection de Maladies Cardiovasculaires",
    initial_sidebar_state="expanded"
)

# --- Définition des colonnes (Basé sur un jeu de données standard) ---
NUM_COLS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CAT_COLS = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
TARGET_COL = 'HeartDisease' # Doit contenir 0 ou 1

# --- 0. Chargement des Données (Avec option d'upload) ---
@st.cache_data(show_spinner="Chargement ou génération des données...")
def load_and_prepare_data(uploaded_file):
    """Charge les données du fichier CSV ou génère des données factices si aucun fichier n'est fourni."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Fichier '{uploaded_file.name}' chargé avec succès. {df.shape[0]} lignes.")
            
            # Vérification des colonnes essentielles
            required_cols = NUM_COLS + CAT_COLS + [TARGET_COL]
            if not all(col in df.columns for col in required_cols):
                st.error("Le fichier CSV ne contient pas les colonnes attendues (HeartDisease, Age, Cholesterol, etc.). Utilisation des données factices en remplacement.")
                df = generate_dummy_data()
            else:
                # S'assurer que la cible est bien binaire et mappée pour la visualisation
                df[TARGET_COL] = df[TARGET_COL].astype(str).str.replace(r'\.0$', '', regex=True).astype(int)
        except Exception as e:
            st.error(f"Erreur de chargement ou de conversion des données : {e}. Utilisation des données factices.")
            df = generate_dummy_data()
    else:
        df = generate_dummy_data()
    
    # Mappage de la variable cible pour une meilleure visualisation
    df[TARGET_COL] = df[TARGET_COL].map({1: 'Malade', 0: 'Sain'})
    
    return df

def generate_dummy_data():
    """Crée des données factices simulant un jeu de données de maladie cardiaque (pour la démo)."""
    N = 918
    age = np.random.randint(20, 80, N)
    resting_bp = np.random.randint(90, 200, N)
    cholesterol = np.random.randint(100, 400, N)
    max_hr = np.random.randint(60, 200, N)
    
    risk_score = (age / 80) + (cholesterol / 400) + (resting_bp / 200) + (np.random.rand(N) * 0.5)
    heart_disease = (risk_score > np.percentile(risk_score, 45)).astype(int) 

    data = {
        'Age': age,
        'Sex': np.random.choice(['M', 'F'], N, p=[0.7, 0.3]),
        'ChestPainType': np.random.choice(['ATA', 'ASY', 'NAP', 'TA'], N),
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': np.random.choice([0, 1], N, p=[0.8, 0.2]),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], N),
        'MaxHR': max_hr,
        'ExerciseAngina': np.random.choice(['Y', 'N'], N),
        'Oldpeak': np.random.uniform(0.0, 4.0, N),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], N),
        'HeartDisease': heart_disease
    }
    st.sidebar.warning("Aucun fichier 'heart.csv' chargé. Utilisation des données factices.")
    return pd.DataFrame(data)

# --- Barre latérale pour l'Upload ---
st.sidebar.title("Configuration des Données")
uploaded_file = st.sidebar.file_uploader("Veuillez charger 'heart.csv'", type=["csv"]) # C'est ici que vous chargez votre fichier

df = load_and_prepare_data(uploaded_file)

st.title("🩺 Pipeline de Détection des Maladies Cardiovasculaires (MCV)")
st.caption("Ce tableau de bord simule un projet de Machine Learning complet en 9 étapes.")


# --- Structure du Tableau de Bord par Onglets ---
tab1, tab2, tab3, tab4, tab56, tab7, tab8, tab9 = st.tabs([
    "1. Exploration des Données", 
    "2. Visualisation (Distributions)", 
    "3. Matrice de Corrélation", 
    "4. Prétraitement des Données", 
    "5 Modélisation et Évaluation", 
    "7. Visualisation des Résultats (ROC)", 
    "8. Optimisation (Random Forest)", 
    "9. Conclusion"
])

# ==============================================================================
# ÉTAPE 1 : EXPLORATION DES DONNÉES
# ==============================================================================
with tab1:
    st.header("Exploration des Données (EDA)")
    st.markdown("Aperçu général du jeu de données, de sa structure et des types de variables.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Dimensions")
        st.info(f"Lignes: **{df.shape[0]}** | Colonnes: **{df.shape[1]}**")
    with col2:
        st.subheader("Variables Numériques")
        st.info(f"Total: **{len(NUM_COLS)}** ({', '.join(NUM_COLS)})")
    with col3:
        st.subheader("Variables Catégorielles")
        st.info(f"Total: **{len(CAT_COLS)}** ({', '.join(CAT_COLS)})")

    st.subheader("Aperçu du DataFrame")
    st.dataframe(df.head())

    st.subheader("Statistiques Descriptives des Variables Numériques")
    st.dataframe(df[NUM_COLS].describe().T)

# ==============================================================================
# ÉTAPE 2 : VISUALISATION 1 - DISTRIBUTIONS
# ==============================================================================
with tab2:
    st.header("Visualisation 1 - Distribution des Variables")
    st.markdown("Analyse de la variable cible et des distributions clés.")

    # Distribution de la variable cible (HeartDisease)
    st.subheader("Distribution de la Variable Cible (HeartDisease)")
    fig_target = px.pie(
        df, 
        names=TARGET_COL, 
        title='Répartition des Cas de Maladie Cardiovasculaire',
        color_discrete_sequence=['red', 'blue']
    )
    st.plotly_chart(fig_target, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Distribution par Âge
    with col_a:
        st.subheader("Distribution par Âge et État de Santé")
        fig_age = px.histogram(
            df, 
            x='Age', 
            color=TARGET_COL, 
            marginal="box", 
            nbins=30, 
            title="Distribution de l'Âge par État MCV",
            color_discrete_map={'Malade': 'red', 'Sain': 'blue'}
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # Distribution par Sexe
    with col_b:
        st.subheader("Distribution par Sexe")
        sex_counts = df.groupby(['Sex', TARGET_COL]).size().reset_index(name='Count')
        fig_sex = px.bar(
            sex_counts, 
            x='Sex', 
            y='Count', 
            color=TARGET_COL, 
            barmode='group', 
            title="Cas de MCV par Sexe",
            color_discrete_map={'Malade': 'red', 'Sain': 'blue'}
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    # Distribution par Type de Douleur Thoracique
    st.subheader("Distribution par Type de Douleur Thoracique")
    cp_counts = df.groupby(['ChestPainType', TARGET_COL]).size().reset_index(name='Count')
    fig_cp = px.bar(
        cp_counts, 
        x='ChestPainType', 
        y='Count', 
        color=TARGET_COL, 
        barmode='group', 
        title="Cas de MCV par Type de Douleur Thoracique",
        color_discrete_map={'Malade': 'red', 'Sain': 'blue'}
    )
    st.plotly_chart(fig_cp, use_container_width=True)


# ==============================================================================
# ÉTAPE 3 : MATRICE DE CORRÉLATION
# ==============================================================================
with tab3:
    st.header("Matrice de Corrélation")
    st.markdown("Visualisation des relations linéaires entre toutes les variables (après encodage simple pour la corrélation).")

    # Préparation des données pour la corrélation (encodage simple)
    df_corr = df.copy()
    # Convertir la cible et les colonnes catégorielles en numérique pour le calcul
    df_corr[TARGET_COL] = df_corr[TARGET_COL].map({'Malade': 1, 'Sain': 0})
    # Encodage des colonnes catégorielles
    df_corr = pd.get_dummies(df_corr, columns=CAT_COLS, drop_first=True)
    
    st.subheader("Heatmap de Corrélation (Numérique + Catégorielle Encodée)")
    
    try:
        corr = df_corr.corr()
        # Filtrer pour ne montrer que les corrélations > |0.1| pour plus de clarté
        corr_filtered = corr[corr.abs() > 0.1]
        
        fig_corr, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_filtered, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=.5,
            cbar_kws={'label': 'Coefficient de Corrélation'},
            ax=ax
        )
        ax.set_title("Matrice de Corrélation (Coefficients > |0.1|)")
        st.pyplot(fig_corr)
    except Exception as e:
        st.error(f"Erreur lors du calcul de la corrélation : {e}")

# ==============================================================================
# ÉTAPE 4 : PRÉTRAITEMENT DES DONNÉES
# ==============================================================================
@st.cache_data(show_spinner="Prétraitement des données en cours...")
def preprocess_data(data):
    """Effectue le prétraitement: Encodage et Mise à l'échelle."""
    # Retirer les colonnes non nécessaires si elles sont apparues
    try:
        X = data.drop(TARGET_COL, axis=1)
        # Assurer que y est la version numérique pour l'entraînement
        y = data[TARGET_COL].map({'Malade': 1, 'Sain': 0}) 
    except Exception as e:
        st.error(f"Erreur de préparation des données pour l'entraînement : {e}")
        return pd.DataFrame(), pd.Series(), None

    # Définition du préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS)
        ],
        remainder='passthrough'
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    # Récupérer les noms de colonnes pour X_processed
    feature_names = preprocessor.get_feature_names_out()
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    return X_processed_df, y, preprocessor

# ==============================================================================
# ÉTAPE 5 & 6 : CONSTRUCTION, ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ==============================================================================
@st.cache_data(show_spinner="Entraînement et évaluation des modèles...")
def train_and_evaluate_models(X, y):
    """Sépare les données, entraîne plusieurs modèles et retourne les résultats."""
    if X.empty or y.empty:
        return pd.DataFrame(), [], None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        results.append({
            'Modèle': name,
            'Accuracy': accuracy,
            'ROC AUC': roc_auc,
            'Temps d\'Entraînement (s)': end_time - start_time,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
            'Model Instance': model
        })

    results_df = pd.DataFrame([
        {'Modèle': r['Modèle'], 'Accuracy': f"{r['Accuracy']:.4f}", 'ROC AUC': f"{r['ROC AUC']:.4f}", 'Temps (s)': f"{r['Temps d\'Entraînement (s)']:.4f}"} 
        for r in results
    ])
    
    return results_df, results, X_test, y_test, X_train, y_train

# Application des étapes 4, 5 et 6 
X_processed, y, preprocessor = preprocess_data(df)
results_df, results_data, X_test, y_test, X_train, y_train = train_and_evaluate_models(X_processed, y)


with tab4:
    st.header(" Prétraitement des Données")
    st.markdown("Préparation du jeu de données pour l'entraînement : Standardisation des variables numériques et Encodage One-Hot des variables catégorielles.")
    
    st.subheader("Transformations Appliquées")
    st.write("Variables Numériques:", NUM_COLS, "→ **StandardScaler** (Mise à l'échelle)")
    st.write("Variables Catégorielles:", CAT_COLS, "→ **OneHotEncoder** (Création de variables binaires)")
    
    if not X_processed.empty:
        st.subheader("Statistiques Post-Prétraitement")
        st.info(f"Dimensions de l'ensemble de fonctionnalités (X) après encodage: **{X_processed.shape}**")
        st.info(f"Dimensions de la variable cible (y): **{y.shape}**")
        
        with st.expander("Aperçu des Données Prétraitées (5 premières lignes)"):
            st.dataframe(X_processed.head())
    else:
        st.warning("Prétraitement impossible. Veuillez vérifier le chargement des données.")


with tab56:
    st.header("Construction, Entraînement et Évaluation des Modèles")
    st.markdown("Quatre modèles de classification ont été entraînés et évalués sur la précision (Accuracy) et l'aire sous la courbe ROC (ROC AUC).")
    
    if not results_df.empty:
        st.subheader("Performance des Modèles de Classification")
        
        st.dataframe(results_df.sort_values(by='ROC AUC', ascending=False).reset_index(drop=True))
        
        # Affichage du rapport de classification du meilleur modèle
        best_model_name = results_df.sort_values(by='ROC AUC', ascending=False).iloc[0]['Modèle']
        best_result = next(r for r in results_data if r['Modèle'] == best_model_name)
        
        st.subheader(f"Rapport de Classification Détaillé pour : {best_model_name}")
        st.text(classification_report(y_test, best_result['Model Instance'].predict(X_test), target_names=['Sain (0)', 'Malade (1)'], zero_division=0))

        # Matrice de confusion
        st.subheader("Matrice de Confusion du Meilleur Modèle")
        cm = confusion_matrix(y_test, best_result['Model Instance'].predict(X_test))
        fig_cm, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Sain (0)', 'Malade (1)'], yticklabels=['Sain (0)', 'Malade (1)'], ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Valeur Réelle')
        st.pyplot(fig_cm)
    else:
        st.warning("Aucun résultat d'entraînement. Veuillez vérifier le prétraitement des données.")

# ==============================================================================
# ÉTAPE 7 : VISUALISATION DES RÉSULTATS (Courbes ROC)
# ==============================================================================
with tab7:
    st.header("Visualisation des Résultats - Courbes ROC")
    st.markdown("La courbe ROC (Receiver Operating Characteristic) et l'AUC montrent la capacité de chaque modèle à distinguer les cas 'Malade' des cas 'Sain'.")
    
    if results_data and not results_df.empty:
        fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
        
        # Tracé de la ligne de base (aléatoire)
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.50)')

        # Tracé des courbes pour chaque modèle
        for r in results_data:
            ax_roc.plot(r['fpr'], r['tpr'], label=f"{r['Modèle']} (AUC = {r['ROC AUC']:.4f})")
        
        ax_roc.set_xlabel('Taux de Faux Positifs (FPR)')
        ax_roc.set_ylabel('Taux de Vrais Positifs (TPR)')
        ax_roc.set_title('Courbes ROC Multi-Modèles')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)
        st.pyplot(fig_roc)
    else:
        st.warning("Impossible de tracer les courbes ROC. Vérifiez les étapes précédentes.")

# ==============================================================================
# ÉTAPE 8 : OPTIMISATION DU MEILLEUR MODÈLE (Random Forest)
# ==============================================================================
@st.cache_data(show_spinner="Optimisation du Random Forest par GridSearchCV...")
def optimize_random_forest(X_train, y_train):
    """Optimisation des hyperparamètres du Random Forest."""
    rf_model = RandomForestClassifier(random_state=42)
    
    # Espace de recherche réduit pour un temps d'exécution rapide dans Streamlit
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

with tab8:
    st.header(" Optimisation du Meilleur Modèle (Random Forest)")
    st.markdown("Le modèle Random Forest est optimisé en utilisant `GridSearchCV` pour trouver la meilleure combinaison d'hyperparamètres.")
    
    if not X_train.empty:
        best_rf_model, best_params, best_score = optimize_random_forest(X_train, y_train)

        st.subheader("Meilleurs Hyperparamètres Trouvés")
        st.json(best_params)
        
        st.subheader("Score (ROC AUC) du Modèle Optimisé")
        st.success(f"ROC AUC sur l'ensemble d'entraînement (Cross-Validation) : **{best_score:.4f}**")
        
        # Évaluation sur l'ensemble de test avec le modèle optimisé
        y_pred_opt = best_rf_model.predict(X_test)
        y_proba_opt = best_rf_model.predict_proba(X_test)[:, 1]
        accuracy_opt = accuracy_score(y_test, y_pred_opt)
        roc_auc_opt = auc(*roc_curve(y_test, y_proba_opt)[:2])

        st.subheader("Performance sur l'Ensemble de Test (Modèle Optimisé)")
        col_opt_1, col_opt_2 = st.columns(2)
        col_opt_1.metric("Accuracy Optimisée", f"{accuracy_opt:.4f}")
        col_opt_2.metric("ROC AUC Optimisé", f"{roc_auc_opt:.4f}")

        with st.expander("Rapport de Classification du Modèle Optimisé"):
            st.text(classification_report(y_test, y_pred_opt, target_names=['Sain (0)', 'Malade (1)'], zero_division=0))
    else:
        st.warning("Impossible de procéder à l'optimisation. Vérifiez l'entraînement des modèles.")

# ==============================================================================
# ÉTAPE 9 : CONCLUSION
# ==============================================================================
with tab9:
    st.header(" Conclusion du Projet")
    st.markdown("Synthèse des résultats et prochaines étapes suggérées.")
    
    if not results_df.empty:
        best_model_name_final = results_df.sort_values(by='ROC AUC', ascending=False).iloc[0]['Modèle']
        roc_auc_max = float(results_df['ROC AUC'].max())
        
        st.info(f"""
            ### Récapitulatif
            * **Données** : Utilisation des données {'chargées' if uploaded_file else 'factices générées'}.
            * **Objectif** : Détecter la maladie cardiovasculaire (MCV).
            * **Meilleur Modèle Initial** : Le **{best_model_name_final}** a montré la meilleure performance avec un ROC AUC de **{roc_auc_max:.4f}**.
            * **Optimisation** : Après l'optimisation des hyperparamètres du Random Forest (si l'étape 8 a été exécutée), le modèle a atteint un ROC AUC sur l'ensemble de test de **{roc_auc_opt:.4f}**.
            
            ### Prochaines Étapes
            1.  **Ingénierie de Fonctionnalités (Feature Engineering)** : Créer des variables plus prédictives, comme l'indice de masse corporelle (IMC).
            2.  **Validation Externe** : Tester le modèle optimisé sur un jeu de données externe (non vu) pour confirmer sa généralisation.
            3.  **Interprétabilité (SHAP/LIME)** : Comprendre quelles variables (âge, cholestérol, etc.) contribuent le plus à chaque prédiction.
        """)
    else:
        st.warning("Synthèse impossible. Veuillez vous assurer que les données ont été chargées et les modèles entraînés.")