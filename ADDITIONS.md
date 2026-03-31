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
