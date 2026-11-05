# E-Commerce Intelligence & Customer Analytics Platform

> **Guide de réalisation --- parcours intensif de 3 jours**\
> Stack : Python, Jupyter, Pandas, NumPy, Matplotlib/Seaborn, SQL,
> scikit-learn, SciPy/Statsmodels, FastAPI, Streamlit, MLflow, Docker et
> Git.

## 1. Présentation du projet

### 1.1 Objectif métier

Construire un prototype de plateforme d'analyse e-commerce qui
transforme des données transactionnelles en informations utiles à la
décision :

-   comprendre les ventes et le comportement des clients ;
-   identifier des segments de clients ;
-   explorer des signaux de risque d'inactivité ;
-   mesurer l'effet potentiel d'une action avec un test A/B ;
-   exposer un modèle via une API et présenter les résultats dans un
    tableau de bord.

Le projet doit rester **honnête sur ce que les données permettent
réellement de conclure**. Un modèle ne prouve pas une causalité, une
segmentation n'est pas une prédiction et un test A/B simulé n'est pas
une expérience réelle.

### 1.2 Dataset conseillé

**UCI Online Retail II** : transactions d'un détaillant en ligne. Il
contient notamment des identifiants de facture, produits, quantités,
dates, prix unitaires et identifiants clients.

Source : https://archive.ics.uci.edu/dataset/502/online+retail+ii

Le dataset est transactionnel : il ne contient pas nécessairement une
colonne fiable `churn` ni une table de sessions web avec visiteurs et
conversions. Par conséquent :

-   la segmentation RFM est adaptée ;
-   l'analyse des ventes est adaptée ;
-   un vrai taux de conversion web ne peut pas être calculé sans données
    de visites/sessions ;
-   une analyse d'inactivité client peut être construite comme
    **proxy**, avec une définition explicite ;
-   l'A/B testing doit être présenté comme un exercice simulé si les
    données d'expérimentation ne sont pas disponibles.

## 2. Périmètre et livrables

### Livrables attendus

1.  `01_data_quality.ipynb` --- chargement, audit et nettoyage.
2.  `02_eda.ipynb` --- analyse exploratoire et KPI.
3.  `03_rfm_segmentation.ipynb` --- création RFM et segmentation
    K-Means.
4.  `04_inactivity_model.ipynb` --- optionnel : définition d'une cible
    proxy et modèle prédictif sans fuite de données.
5.  `05_ab_testing.ipynb` --- simulation d'un test A/B et interprétation
    statistique.
6.  Une API FastAPI qui charge un modèle entraîné et renvoie une
    prédiction.
7.  Un tableau de bord Streamlit pour visualiser les KPI et segments.
8.  Un suivi d'expérience MLflow si le temps le permet.
9.  Un `README.md`, un `requirements.txt`, des tests simples et un dépôt
    Git.

### Ce qui est réaliste en trois jours

Le but des trois jours est de produire un **prototype propre et
reproductible**, pas un système de production complet. Priorité :
qualité des données, EDA, RFM/K-Means et interprétation. Le modèle
prédictif et le déploiement sont des extensions si le temps et les
données le permettent.

## 3. Organisation du dépôt

``` text
ecommerce-intelligence/
├── data/
│   ├── raw/                 # Données originales, non modifiées
│   ├── processed/            # Données nettoyées et agrégées
│   └── README.md             # Source, licence, dictionnaire et provenance
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_rfm_segmentation.ipynb
│   ├── 04_inactivity_model.ipynb
│   └── 05_ab_testing.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── rfm.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── evaluation/
│       └── metrics.py
├── app/
│   ├── api.py
│   └── dashboard.py
├── models/                   # Artefacts locaux, à ne pas versionner si volumineux
├── tests/
│   └── test_preprocessing.py
├── reports/
│   └── figures/
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

**Bonnes pratiques :**

-   conserver les données originales dans `data/raw/` et ne jamais les
    écraser ;
-   ne pas publier de données sensibles ou de fichiers volumineux sur
    GitHub ;
-   ne pas versionner les environnements virtuels, secrets, caches ou
    modèles lourds ;
-   séparer progressivement le code réutilisable (`src/`) des
    expérimentations (`notebooks/`) ;
-   fixer les versions des dépendances dans `requirements.txt` ;
-   utiliser des chemins relatifs et une configuration centralisée ;
-   documenter toute transformation et toute hypothèse métier.

## 4. Installation et préparation

### 4.1 Créer l'environnement

Windows PowerShell :

``` powershell
mkdir ecommerce-intelligence
cd ecommerce-intelligence
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux :

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Installer les bibliothèques

Créer `requirements.txt` :

``` text
pandas
numpy
openpyxl
matplotlib
seaborn
scikit-learn
scipy
statsmodels
jupyter
joblib
fastapi
uvicorn[standard]
streamlit
mlflow
pytest
```

Puis :

``` bash
pip install -r requirements.txt
```

Enregistrer ensuite les versions réellement installées :

``` bash
pip freeze > requirements-lock.txt
```

Pour un projet partagé, vérifier les versions et tester l'installation
dans un environnement propre.

### 4.3 Récupérer les données

Télécharger le fichier depuis la page UCI et le placer dans `data/raw/`.
Ne pas modifier manuellement le fichier source. Vérifier la licence et
les conditions d'utilisation avant toute redistribution.

## 5. Jour 1 --- Qualité des données, EDA et statistiques

### Étape 1 --- Comprendre les données

Dans `01_data_quality.ipynb` :

1.  importer les bibliothèques ;
2.  charger le fichier ;
3.  afficher les premières et dernières lignes ;
4.  vérifier les dimensions, types et noms de colonnes ;
5.  examiner les valeurs manquantes et doublons ;
6.  vérifier les valeurs inhabituelles.

Exemple :

``` python
import pandas as pd

df = pd.read_excel("data/raw/online_retail_II.xlsx")

print(df.shape)
display(df.head())
df.info()
display(df.isna().sum().sort_values(ascending=False))
display(df.describe(include="all").T)
```

Les noms exacts des colonnes et le nom du fichier peuvent varier selon
la version téléchargée. Les vérifier avant de poursuivre.

### Étape 2 --- Définir les règles de nettoyage

Examiner notamment :

-   lignes sans identifiant client ;
-   quantités nulles ou négatives ;
-   prix unitaires nuls ou négatifs ;
-   factures annulées ou retours ;
-   doublons exacts ;
-   dates invalides.

Ne pas supprimer automatiquement toutes les lignes négatives : elles
peuvent représenter des retours ou annulations. Décider d'une règle, la
justifier et conserver une trace du nombre de lignes concernées.

Exemple de préparation minimale :

``` python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

# Exemple de vue pour l'analyse des ventes positives.
sales = df[
    df["Customer ID"].notna()
    & df["InvoiceDate"].notna()
    & (df["Quantity"] > 0)
    & (df["Price"] > 0)
].copy()

sales["Revenue"] = sales["Quantity"] * sales["Price"]
```

Cette vue est une règle analytique pour étudier les ventes positives ;
elle ne signifie pas que les retours n'existent pas. Les analyser
séparément si cela est pertinent.

### Étape 3 --- Analyse exploratoire (EDA)

Questions métier à traiter :

-   Quel est le chiffre d'affaires par mois ?
-   Quels produits ou catégories génèrent le plus de ventes, si la
    catégorie existe ?
-   Combien de clients distincts observe-t-on ?
-   Quelle est la distribution du montant des commandes ?
-   Quels pays représentent le plus de ventes ?
-   Observe-t-on une saisonnalité ?

Exemple de KPI :

``` python
total_revenue = sales["Revenue"].sum()
number_of_orders = sales["Invoice"].nunique()
number_of_customers = sales["Customer ID"].nunique()
average_order_value = total_revenue / number_of_orders

print({
    "revenue": total_revenue,
    "orders": number_of_orders,
    "customers": number_of_customers,
    "average_order_value": average_order_value,
})
```

Préciser la définition de chaque KPI. Ici, le panier moyen est le
chiffre d'affaires positif divisé par le nombre de factures distinctes
de la vue filtrée.

Créer quelques graphiques lisibles : ventes mensuelles, top produits,
distribution des commandes et ventes par pays. Chaque graphique doit
avoir un titre, des axes nommés, des unités et une courte
interprétation.

### Étape 4 --- Statistiques descriptives

Maîtriser :

-   moyenne, médiane, mode ;
-   variance et écart-type ;
-   quantiles et valeurs aberrantes ;
-   corrélation et différence entre corrélation et causalité ;
-   distributions et échantillonnage.

**Livrable Jour 1 :** un notebook reproductible, un dataset nettoyé et
un court résumé des principaux constats.

## 6. Jour 2 --- RFM, segmentation et Machine Learning

### Étape 1 --- Construire les variables RFM

RFM signifie :

-   **Recency** : nombre de jours depuis le dernier achat ;
-   **Frequency** : nombre de commandes distinctes ;
-   **Monetary** : montant total dépensé.

Définir une date de référence fixe, idéalement postérieure à la dernière
date d'achat du jeu de données :

``` python
reference_date = sales["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = sales.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("Revenue", "sum"),
).reset_index()
```

Vérifier les distributions. RFM peut être très asymétrique ; envisager
une transformation logarithmique pour Frequency et Monetary si elle est
justifiée.

### Étape 2 --- Préparer les variables

-   exclure l'identifiant client des variables d'entraînement ;
-   traiter les valeurs manquantes ;
-   examiner les distributions et valeurs extrêmes ;
-   mettre les variables à l'échelle avant K-Means ;
-   ajuster le scaler uniquement sur les données d'entraînement
    lorsqu'il y a un découpage train/test.

Exemple de mise à l'échelle pour une segmentation exploratoire :

``` python
from sklearn.preprocessing import StandardScaler

features = rfm[["Recency", "Frequency", "Monetary"]].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

Pour des variables très asymétriques, tester une transformation adaptée
avant la standardisation et comparer les résultats.

### Étape 3 --- Appliquer K-Means

K-Means regroupe les clients en K groupes selon leur proximité dans
l'espace des variables.

``` python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
rfm["Cluster"] = kmeans.fit_predict(X_scaled)
```

Ne pas choisir `K=4` sans justification. Tester plusieurs valeurs de K,
utiliser la méthode du coude et le score de silhouette, puis examiner si
les groupes sont interprétables d'un point de vue métier.

Créer un profil par cluster :

``` python
cluster_profile = rfm.groupby("Cluster")[
    ["Recency", "Frequency", "Monetary"]
].mean().round(2)

display(cluster_profile)
```

Donner des noms aux segments seulement après avoir observé leurs
caractéristiques, par exemple « clients fréquents » ou « clients à
réactiver ».

### Étape 4 --- Comprendre les modèles supervisés

La classification nécessite une **variable cible** (label) fiable.
Online Retail II ne fournit pas directement une étiquette de churn.

Option d'extension : définir un proxy d'inactivité à partir d'une date
de coupure temporelle :

1.  choisir une date de coupure `T` ;
2.  calculer les variables client uniquement avec les transactions
    antérieures ou égales à `T` ;
3.  définir une fenêtre future de durée `H` ;
4.  définir la cible comme absence d'achat pendant `H` jours après `T` ;
5.  exclure les clients pour lesquels toute la fenêtre future n'est pas
    observable ;
6.  séparer les périodes temporellement pour éviter la fuite de données.

Cette cible est une **définition opérationnelle d'inactivité**, pas une
vérité universelle sur le churn. Documenter `T`, `H` et la règle de
labellisation.

Pour le modèle :

-   commencer par une baseline simple ;
-   comparer Logistic Regression et Random Forest ;
-   utiliser un découpage temporel lorsque la cible dépend du temps ;
-   construire le preprocessing dans un `Pipeline` ;
-   évaluer avec Precision, Recall, F1-score, ROC-AUC et matrice de
    confusion ;
-   comparer au modèle baseline et expliquer les erreurs.

Exemple de pipeline (à adapter aux variables et à la cible réellement
construites) :

``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```

Pour un vrai workflow, entraîner et évaluer le pipeline sur des
partitions séparées. Ne jamais ajuster le scaler ou sélectionner les
variables sur l'ensemble des données avant le split : cela crée une
fuite de données.

### Étape 5 --- Métriques et diagnostic

-   **Accuracy** : proportion de prédictions correctes ; peut être
    trompeuse si les classes sont déséquilibrées.
-   **Precision** : parmi les clients prédits positifs, part réellement
    positive.
-   **Recall** : parmi les vrais positifs, part détectée.
-   **F1-score** : compromis entre precision et recall.
-   **ROC-AUC** : capacité de classement sur différents seuils ;
    compléter avec PR-AUC si la classe positive est rare.

Expliquer quelle erreur coûte le plus à l'entreprise avant de choisir un
seuil.

**Livrable Jour 2 :** segmentation interprétée et, si la cible proxy est
correctement définie, premier modèle évalué sans fuite de données.

## 7. Jour 3 --- A/B testing, API, dashboard et MLOps

### Étape 1 --- A/B testing

Un test A/B compare un groupe témoin A à un groupe exposé à une variante
B. Il faut une assignation aléatoire, une métrique définie à l'avance,
une taille d'échantillon suffisante et une durée adaptée.

Pour un test de conversion, il faut des données de visiteurs exposés et
de conversions. Le dataset Online Retail II ne fournit pas directement
ces événements web. Réaliser donc une **simulation pédagogique** ou
trouver un dataset d'expérimentation adapté ; ne pas présenter des
résultats simulés comme une expérience réelle.

À maîtriser :

-   hypothèse nulle H0 et hypothèse alternative H1 ;
-   seuil alpha ;
-   p-value ;
-   erreur de type I et type II ;
-   puissance statistique ;
-   taille d'effet et intervalle de confiance.

La significativité statistique ne garantit pas un impact business
important. Rapporter aussi la taille d'effet et son incertitude.

### Étape 2 --- Sauvegarder un modèle

Sauvegarder le pipeline complet (prétraitement + modèle), pas seulement
le classifieur :

``` python
import joblib

joblib.dump(model, "models/inactivity_pipeline.joblib")
```

Charger uniquement des artefacts de confiance. Les fichiers
pickle/joblib peuvent exécuter du code lors du chargement.

### Étape 3 --- Créer une API FastAPI

Exemple minimal à placer dans `app/api.py`. Adapter les champs aux
variables réellement utilisées par le pipeline :

``` python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="E-Commerce Customer Analytics API")
model = joblib.load("models/inactivity_pipeline.joblib")

class CustomerFeatures(BaseModel):
    Recency: float = Field(ge=0)
    Frequency: float = Field(ge=0)
    Monetary: float = Field(ge=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: CustomerFeatures):
    X = pd.DataFrame([features.model_dump()])
    prediction = int(model.predict(X)[0])
    return {"prediction": prediction}
```

Démarrer depuis la racine du dépôt :

``` bash
uvicorn app.api:app --reload
```

Consulter `http://127.0.0.1:8000/docs` pour tester l'API. En production,
ajouter gestion des erreurs, authentification si nécessaire,
journalisation, validation stricte, tests et configuration sécurisée. Ne
pas exposer de données personnelles dans les logs.

### Étape 4 --- Construire un dashboard Streamlit

Le dashboard peut contenir :

-   KPI de ventes ;
-   évolution mensuelle du chiffre d'affaires ;
-   top produits et pays ;
-   répartition des segments RFM ;
-   profil moyen de chaque cluster ;
-   formulaire de prédiction si le modèle supervisé existe.

L'interface doit afficher les limites du modèle et la définition des
indicateurs. Éviter de présenter une prédiction comme une certitude.

Lancement :

``` bash
streamlit run app/dashboard.py
```

### Étape 5 --- Suivre les expériences avec MLflow

Enregistrer pour chaque essai :

-   paramètres (algorithme, hyperparamètres, variables) ;
-   métriques (F1, Recall, ROC-AUC, etc.) ;
-   artefacts utiles (matrice de confusion, graphiques) ;
-   version du code ou identifiant du commit ;
-   description de la cible et du split.

Exemple :

``` python
import mlflow
from sklearn.metrics import f1_score

with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
```

Ne pas enregistrer de données personnelles ou de secrets comme
artefacts.

### Étape 6 --- Docker (extension)

Créer un `Dockerfile` adapté à l'application. Vérifier que le conteneur
démarre, que les dépendances sont installées et que les fichiers modèle
nécessaires sont disponibles. Ne pas inclure de secrets dans l'image.
Utiliser une image de base maintenue et épingler les versions pour une
exécution reproductible.

**Livrable Jour 3 :** simulation A/B documentée, API testable, dashboard
minimal et suivi MLflow. Docker peut être finalisé après les trois
jours.

## 8. Bonnes pratiques Data Science à respecter

### Qualité et reproductibilité

-   définir les objectifs métier avant de choisir un algorithme ;
-   inspecter les données avant de les nettoyer ;
-   documenter les règles de nettoyage et les exclusions ;
-   fixer les graines aléatoires lorsque cela est utile ;
-   séparer données brutes, transformées et résultats ;
-   rendre les notebooks exécutables de haut en bas ;
-   déplacer le code réutilisable dans `src/` ;
-   enregistrer les dépendances et les versions.

### Éviter les erreurs fréquentes

-   **Data leakage** : ne jamais utiliser des informations futures pour
    prédire le passé ;
-   **Mauvais split** : utiliser un split temporel lorsque la prédiction
    porte sur le futur ;
-   **Métrique inadaptée** : ne pas se limiter à l'accuracy en cas de
    déséquilibre ;
-   **Surinterprétation** : corrélation ne signifie pas causalité ;
-   **K-Means mal interprété** : les clusters sont des groupes
    mathématiques, pas automatiquement des personas métier ;
-   **A/B test non valide** : une comparaison de deux groupes non
    randomisés ne prouve pas l'effet causal ;
-   **Mauvaise cible** : ne pas appeler « churn réel » une simple règle
    d'inactivité sans la documenter.

### Sécurité et Git

Exemple de `.gitignore` :

``` gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
.env
*.pyc
data/raw/*
data/processed/*
models/*.joblib
mlruns/
```

Ajouter une exception si certains petits fichiers de données ou
artefacts peuvent être partagés légalement et sans risque. Ne jamais
committer de mots de passe, clés API ou fichiers `.env`.

Commits clairs et fréquents :

``` bash
git add .
git commit -m "Add initial data quality analysis"
git commit -m "Add exploratory sales analysis"
git commit -m "Add RFM customer segmentation"
```

## 9. Critères de validation du projet

Avant de considérer le prototype terminé, vérifier :

-   [ ] La source et la structure des données sont documentées.
-   [ ] Les règles de nettoyage sont expliquées.
-   [ ] Les KPI ont des définitions précises.
-   [ ] Les graphiques sont lisibles et interprétés.
-   [ ] Les variables RFM sont correctement calculées.
-   [ ] Le choix de K est justifié et les clusters interprétés.
-   [ ] Toute cible supervisée est définie et documentée.
-   [ ] Le split empêche la fuite de données.
-   [ ] Les métriques sont adaptées au problème.
-   [ ] Le test A/B est réel ou clairement identifié comme simulation.
-   [ ] L'API valide les entrées et charge le pipeline complet.
-   [ ] Le dashboard présente des résultats cohérents.
-   [ ] Le README explique installation, exécution, données, limites et
    résultats.
-   [ ] Aucun secret ou fichier sensible n'est publié.

## 10. Proposition de présentation dans le CV

À utiliser uniquement après avoir réellement implémenté et vérifié les
éléments correspondants :

**E-Commerce Intelligence & Customer Analytics Platform** --- Python,
Pandas, Scikit-learn, SQL, FastAPI, Streamlit, MLflow

-   Performed data cleaning, exploratory analysis and sales KPI
    reporting on transactional e-commerce data.
-   Engineered RFM customer features and applied K-Means clustering to
    identify customer segments.
-   Developed and evaluated a customer inactivity prediction pipeline
    using leakage-aware validation (if implemented).
-   Built a Streamlit dashboard and exposed model predictions through a
    FastAPI endpoint (if implemented).
-   Tracked model experiments and evaluation metrics with MLflow (if
    implemented).

## 11. Ressources officielles

-   Dataset UCI Online Retail II :
    https://archive.ics.uci.edu/dataset/502/online+retail+ii
-   Kaggle Learn : https://www.kaggle.com/learn
-   Pandas : https://pandas.pydata.org/docs/
-   Scikit-learn : https://scikit-learn.org/stable/
-   Google Machine Learning Crash Course :
    https://developers.google.com/machine-learning/crash-course
-   StatQuest : https://statquest.org/
-   Statsmodels : https://www.statsmodels.org/
-   FastAPI : https://fastapi.tiangolo.com/
-   Streamlit : https://docs.streamlit.io/
-   MLflow : https://mlflow.org/docs/latest/
-   Docker : https://docs.docker.com/get-started/

------------------------------------------------------------------------

**Conseil final :** en trois jours, privilégier un pipeline de données
fiable, une analyse bien expliquée et une segmentation RFM correcte
plutôt que d'ajouter beaucoup d'outils superficiellement. Un projet
simple, reproductible et bien défendu en entretien est plus convaincant
qu'une longue liste de technologies seulement mentionnées.
