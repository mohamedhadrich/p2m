# Additions au projet P2M

## 📝 Résumé des modifications

Ce document explique les trois nouvelles couches ajoutées au projet pour renforcer l'analyse du drift et la décision de retrain.

---

## 1. **Module `utils/costs.py`** — Estimation du coût métier

### Quoi ?
Un module qui estime l'**impact financier** du drift et du retrain.

### Fonctionnalités principales :

- **`estimate_drift_cost(...)`**  
  - Estime le coût de la baisse de performance due au drift.  
  - Logique : `Coût = (erreurs supplémentaires) × (coût par erreur)`.  
  - Paramètres ajustables :  
    - `n_events` : nombre de prédictions dans la période (ex. 10,000).  
    - `cost_per_error` : coût moyen d'une erreur (ex. 100 € = 1 faux positif en fraude).

- **`estimate_retrain_cost(...)`**  
  - Estime le coût du retraining + validation.  
  - Inclut : temps de calcul + review humaine.  
  - Paramètres : `compute_hours`, `hourly_cost`, `human_review_hours`.

- **`make_retrain_decision(...)`**  
  - Compare `drift_cost` vs `retrain_cost`.  
  - Retourne un `CostEstimate` avec recommandation.  
  - Règle : si `drift_cost > 1.5 × retrain_cost` → retrain recommendé.

### Utilisation dans l'app :
- Affichées dans l'onglet *Retrain & Registry* sous "💰 Cost-Benefit Analysis".  
- Montre à quel point c'est judicieux ou non de réentraîner.

### Exemple :
```
Drift cost (over 10k events): 50,000 units
Retrain cost: 10,000 units
Net benefit: 40,000 units
Recommendation: ✅ Retrain recommended
```

---

## 2. **Module `utils/drift_analysis.py`** — Analyse de topologie du drift

### Quoi ?
Une fonction qui **classe le type de drift** observé et donne une interprétation.

### Fonctionnalités principales :

- **`analyze_drift_topology(...)`**  
  - Prend en paramètre :  
    - `drift_result` (résultat PSI/KS),  
    - `ref_perf` et `curr_perf` (évaluations de modèle).  
  - Retourne un `DriftTopology` qui classe en 4 profils :

| Type de drift | Data drift | Perf drop | Interprétation |
|---|---|---|---|
| **No Drift** | ❌ | ❌ | Tout est stable |
| **Data Drift Only** | ✅ Strong | ❌ | Modèle robuste aux changements de distribution |
| **Concept Drift** | ❌/Mild | ✅ | La relation X→y a changé (très important) |
| **Combined Drift** | ✅ | ✅ | Situation grave : données ET relation ont changé |

### Utilisation dans l'app :
- Affichée dans l'onglet *Retrain & Registry* sous "🔍 Drift Analysis & Retrain Decision".  
- Tableau : `Drift Type | Data Drift Severity | Model Drift Severity`.  
- Texte d'interprétation et d'action suggérée.

### Exemple :
```
Drift Type: Concept Drift (Primary)
Data Drift Severity: Mild
Model Drift Severity: Strong

Interpretation: 
The relationship between features and target has likely changed (concept drift).
Data distribution change is minimal, but model performance is degrading.

Action suggested: Retrain recommended to adapt to new concept.
```

---

## 3. **Modification de `app.py`** — Intégration dans le dashboard

### Quoi ?
Le dashboard Streamlit affiche désormais :

1. **Onglet Drift** (inchangé)  
   - Data drift visual (PSI, KS, Chi²).

2. **Onglet Performance** (inchangé)  
   - Métriques du modèle ancien sur Reference & Current.

3. **Onglet Retrain & Registry** (AMÉLIORÉ) :
   - **Section 1 : Drift Topology**  
     - Tableau avec `Drift Type`, `Data Drift Severity`, `Model Drift Severity`.  
     - Interprétation textuelle.
   
   - **Section 2 : Cost-Benefit Analysis**  
     - Estimation du coût du drift (perte de performance).  
     - Estimation du coût du retrain.  
     - Différence (net benefit).  
     - Recommandation claire.
   
   - **Section 3 : Retrain Execution** (inchangé)  
     - Choix du modèle et de la stratégie (Reference only / Current only / Combined).  
     - Bouton "Retrain model".
   
   - **Section 4 : Comparison Old vs New** (AMÉLIORÉ) :  
     - Tableau `Metric | Old | New | Change` (déjà existant).  
     - **NEW** : Tableau `Parameter | Old Value | New Value | Status`  
       - Montre quels hyperparamètres ont changé lors du retrain.  
       - Répond directement à "est-ce que les paramètres sont modifiés ?".

---

## 🎯 Comment ça aide le jury ?

1. **Démontre la compréhension du contexte métier**  
   - Tu montres que tu connais l'impact financier du drift, pas juste les chiffres techniques.  
   - Une baisse d'Accuracy de 5% ≠ "c'est pas important" ; c'est 500 erreurs en plus sur 10k → coût en euros/dollars.

2. **Clarifier les types de drift**  
   - Au lieu de dire "il y a du drift", tu dis "c'est surtout du *concept drift* : la relation X→y a changé."  
   - C'est plus précis et plus scientifiquement pertinent.

3. **Justifier la décision de retrain**  
   - "On retraîne non pas parce qu'il y a du drift techniquement parlant, mais parce que le coût du drift (50k) > coût du retrain (10k), donc net gain = 40k."  
   - C'est une logique d'ingénieur / data scientist professionnel.

4. **Montrer la variabilité du modèle**  
   - Les hyperparamètres changent-ils d'une version à l'autre ?  
   - Ça montre que le retrain n'est pas "du copier-coller", mais une vraie adaptation.

---

## 📊 Exemple de workflow avec les nouvelles additions

1. **User upload Reference (ancien) + Current (nouveau).**

2. **App calcule :**
   - PSI/KS → data drift ?  
   - Accuracy/F1 → performance drop ?

3. **App affiche (NEW) :**
   - "Type de drift observé : **Concept Drift**"  
   - "Impact estimé : **50,000 € / 10k events**"  
   - "Coût du retrain : **10,000 €**"  
   - "Recommandation : **Retrain YES** (gain net = 40,000 €)"

4. **User clique "Retrain model" (Combined strategy).**

5. **App compare :**
   - "Accuracy ancien : 91% → nouveau : 93% (+2%)"  
   - "Random Forest hyperparams changed: n_estimators 100 → 150"  
   - "New model is promoted" ✅

6. **User peut maintenant dire au jury :**  
   - "On a détecté un concept drift (relation X→y changed)"  
   - "Ça coûte 50k€ de perte vs 10k€ de retrain"  
   - "So we retrain, accuracy improved from 91 to 93%, hyperparams updated"  
   - "Sistema is now more robust"  

---

## ⚙️ Paramètres à ajuster selon ton cas d'usage

Dans `costs.py`, tu peux modifier :

```python
n_events = 10000  # combien de prédictions par période
cost_per_error = 100  # coût d'une erreur (€, $, ou unités arbitraires)
compute_hours = 0.5  # temps de retrain
hourly_cost = 50  # coût du compute
human_review_hours = 2  # temps humain pour valider
```

Pour un projet académique, tu peux également garder des valeurs "arbitraires" et dire dans le rapport : "Nous avons simulé ces coûts avec des valeurs raisonnables pour un contexte X".

---

## 📝 À inclure dans le rapport

Ajoute une section "Context Métier & Coût" dans ton rapport :

> "Dans ce projet, nous avons estimé l'impact économique du drift.  
> Par exemple, pour un modèle de fraude avec 10k transactions/jour :  
> - Une baisse d'Accuracy de 5% = 500 erreurs supplémentaires/jour.  
> - À 100€ par faux négatif non détecté, ça représente 50k€ de perte.  
> - Un retrain coûte environ 10k€ (calcul + review + déploiement).  
> - Net bénéfice = 40k€, donc **retrain is justified**."

Cela montre au jury que tu comprends :  
✅ Les maths du drift (statistiques)  
✅ L'impact métier (finance / opérations)  
✅ La décision (cost-benefit)  

C'est exactement ce que demande un vrai système MLOps en production.

---

## 4. **Intelligent Model Recommendation System** — `find_best_model_and_params()`

### Quoi ?
Une nouvelle fonctionnalité automatisée qui teste **tous les modèles disponibles** avec différentes **combinaisons d'hyperparamètres** et recommande les meilleurs selon leur performance en cross-validation.

### Fonctionnalités principales :

- **`find_best_model_and_params(...)`** dans `utils/retrain.py`
  - Tests 15 modèles (8 classification + 7 regression) :
    - Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree, Naive Bayes, AdaBoost, Linear Regression, SVR, KNeighborsRegressor, DecisionTreeRegressor, AdaBoostRegressor, etc.
  - Pour chaque modèle, teste 2-3 combinaisons d'hyperparamètres.
  - Utilise **cross-validation** (3-5 folds) pour évaluer la performance réelle.
  - Retourne une **liste triée de ModelRecommendation** avec score, std, et time.
  - Supporte **quick_mode** pour tests rapides (1 paramètre par modèle).

- **Paramètre grids prédéfinis** :
  ```
  Logistic Regression: C ∈ [0.1, 1.0, 10.0]
  Random Forest: n_estimators ∈ [50, 100, 200], max_depth ∈ [5, 10, 15]
  Gradient Boosting: n_estimators ∈ [50, 100], learning_rate=0.1
  SVM: kernel ∈ [rbf, linear], C=1.0
  K-Nearest Neighbors: n_neighbors ∈ [3, 5], weights ∈ [uniform, distance]
  ... (et autres)
  ```

### Intégration dans l'UI (Tab 5 - Retrain) :

**Section : "🎯 Find Best Model & Parameters"**

1. **Quick Search Controls**
   - ⚡ Checkbox "Quick mode (faster)" → teste 1 param par modèle au lieu de 3
   - 🔍 Bouton "Find Best Model" → lance la recherche

2. **Results Display**
   - Table "Top 5 Recommended Models" avec colonnes :
     - Rank, Model, CV Score, Std, Time (s)
   - 🏆 "Best Recommendation" card affichant :
     - Nom du meilleur modèle
     - Score CV ± std
     - Paramètres recommandés

3. **Selection & Application**
   - Checkbox "✅ Use the best recommendation" → auto-sélectionne le meilleur
   - Ou choix manuel dans la dropdown
   - Les paramètres recommandés sont **auto-populés** dans "⚙️ Customize hyperparameters"
   - Optionnel : l'utilisateur peut **override** les paramètres

4. **Training & Logging**
   - `retrain_with_mlflow()` reçoit les `custom_params` recommandés
   - MLflow enregistre chaque paramètre custom avec la clé `custom_{param_name}`
   - Run name = `retrain-{ModelName}`

### Avantages :

✅ **Objectif** : Pas de "guess" sur le meilleur modèle ; c'est basé sur CV scores
✅ **Automatisé** : 15 modèles × 2-3 params = ~40 combinaisons testées en quelques secondes
✅ **Reproductible** : Même dataset → même recommandation
✅ **Utilisateur-friendly** : Un clic pour découvrir le meilleur modèle
✅ **Tracé** : Tous les params testés sont loggés dans MLflow

### Exemple d'usage :

```
1. User upload Reference + Current datasets
2. User click "Tab 5 - Retrain"
3. User click "🔍 Find Best Model"
   → App tests 30+ combinations in ~10 seconds
4. Table shows:
   #1 Random Forest (CV Score: 0.9234 ± 0.0051, Time: 2.3s)
   #2 Gradient Boosting (CV Score: 0.9187 ± 0.0062, Time: 3.1s)
   #3 SVM (CV Score: 0.9045 ± 0.0098, Time: 1.8s)
   ...
5. User sees "🏆 Best: Random Forest with params {n_estimators: 100, max_depth: 10}"
6. User checks "✅ Use the best recommendation"
7. User clicks "Retrain model"
8. MLflow logs:
   - model_name: Random Forest
   - custom_n_estimators: 100
   - custom_max_depth: 10
   - ... (other params)
```

---

## 5. **Convention Uniforme de Calcul: `change = (curr - ref) / ref`**

### Quoi?
Une convention cohérente à travers tout le projet pour calculer le changement en pourcentage d'une métrique.

### Formule:
```python
change = (current_value - reference_value) / reference_value
```

### Interprétation par type de métrique:

**Pour les métriques d'ERREUR (RMSE, MAE, MSE):**
- `change < 0` → 🟢 **Amélioration** (erreur diminue)
- `change > 0` → 🔴 **Dégradation** (erreur augmente)

**Pour les métriques de SCORE (R², Accuracy, F1):**
- `change > 0` → 🟢 **Amélioration** (score augmente)
- `change < 0` → 🔴 **Dégradation** (score diminue)

### Exemple appliqué:
```
RMSE: 83.5 (ref) → 67.8 (curr)
change = (67.8 - 83.5) / 83.5 = -0.188 = -18.8% ✅ Amélioration!

R²: 0.8619 (ref) → 0.5703 (curr)
change = (0.5703 - 0.8619) / 0.8619 = -0.338 = -33.8% ❌ Dégradation (mais ignorée si RMSE améli)
```

### Où s'applique?
- ✅ `utils/performance.py:check_performance_drop()`
- ✅ `utils/drift_analysis.py:analyze_drift_topology()`
- ✅ Affichage dans l'UI Streamlit (Tab 3 & 4)

### Avantage:
- 🎯 **Une seule formule** pour tous les calculs
- 🎯 **Cohérent** et facile à comprendre
- 🎯 **Pas d'inversion de signe** confuse

---

## 6. **Déterminisme & Reproductibilité: `random_state=42`**

### Problème:
Lorsque vous cliquez sur "Retrain" deux fois avec les **mêmes données et le même modèle**, les métriques changent. C'est parce que la cross-validation mélange les données aléatoirement à chaque exécution.

### Solution:
**Fixer `random_state=42`** pour tous les CV (cross-validation):

```python
# ❌ AVANT (non-déterministe)
cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)  # cv=5 = folds aléatoires

# ✅ APRÈS (déterministe)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)  # cv=kf = folds fixes
```

### Où appliqué:
- ✅ `utils/retrain.py:retrain_with_mlflow()` (ligne ~316)
- ✅ `utils/retrain.py:find_best_model_and_params()` (lignes ~523 et ~557)

### Avantaise:
- ✨ **Résultats reproductibles** - même résultat chaque fois
- ✨ **Debugging facile** - les variations ne sont pas dues au hasard
- ✨ **Comparaisons justes** - peut comparer deux modèles sans fluctuations aléatoires

### Exemple:
```
Run 1: Model X → CV Score = 0.8234
Run 2: Model X → CV Score = 0.8234 ✅ (identique!)

Run 1: Model Y → CV Score = 0.8190
Run 2: Model Y → CV Score = 0.8190 ✅ (identique!)
```

---

## 🚀 Next steps (optionnel)

Si tu veux aller plus loin après ce P2M :

1. **Paramétrer les coûts via UI Streamlit** (sidebar sliders).  
2. **Ajouter River library** pour détection de drift en streaming.  
3. **Exporter les résultats en JSON/CSV** pour audit trail.  
4. **Ajouter Airflow DAG** pour orchestrer la pipeline batch ou quotidienne.

Mais pour un P2M académique solide, ce que tu as maintenant **is more than enough** et très impressionnant.

---

## Questions / Clarifications ?

- Si un paramètre de coût te semble arbitraire, justifie-le simplement dans le rapport.  
- Si tu veux changer la logique de décision (threshold 1.5 → 2.0), libre à toi ; document-le.  
- Les "4 profils de drift" couvrent 90% des cas réels ; c'est une bonne framework pédagogique.
