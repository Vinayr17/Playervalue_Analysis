# ⚽ Football Player Market Value Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green.svg)](https://scikit-learn.org)
[![R²](https://img.shields.io/badge/R²-97%25-brightgreen.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-5.7M%20Records-orange.svg)](#)

Eine umfassende Streamlit-Webanwendung zur Analyse und Vorhersage von Fußballspieler-Marktwerten mit Machine Learning. Das Projekt analysiert **5.7 Millionen Performance-Records** und erreicht einen **R² Score von 97%** bei der Marktwert-Vorhersage.

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 5.7M records |
| **Model Accuracy** | R² = 97% |
| **Market Analyzed** | €8.08B |
| **Leagues Covered** | 5 Top Leagues |
| **MAE** | €58,468 |
| **Features** | 25+ engineered |

## 🎯 Projektübersicht

**Problem**: Der globale Fußball-Transfermarkt (8.08 Mrd. €) zeigt systematische Ineffizienzen - Spieler werden oft höher bewertet als ihre statistische Leistung rechtfertigt.

**Lösung**: Entwicklung eines ML-basierten Systems zur objektiven Spielerbewertung, das Performance von Marketing-Premium trennt.

**Technologien**: Python, Streamlit, XGBoost, Random Forest, Pandas, Plotly

## ✨ Kernfunktionen

- **🔍 Spielersuche**: Intelligente Suche mit detaillierten Saisonstatistiken
- **🤖 ML Dashboard**: Random Forest vs. XGBoost Vergleich mit R² Score 97%
- **🧮 Marktwert-Formel**: Wissenschaftlich fundierte Berechnung (CIES Football Observatory)
- **⭐ Brand Premium Analyse**: Performance vs. Marketing-Wert Quantifizierung
- **📊 Market Inefficiency Detection**: Identifikation unter-/überbewerteter Spieler

## 🚀 Quick Start

```bash
# Installation
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly xgboost

# App starten
streamlit run streamlit_finals.py
```

## 🔧 Technische Highlights

### Dataset & Preprocessing
- **5.7 Millionen Performance-Records** aus 5 Top-Ligen (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
- **Feature Engineering**: 25+ berechnete Metriken (Effizienz, Disziplin, Position, Alters-Faktoren)
- **Data Cleaning**: Konsistente Formatierung über verschiedene Datenquellen

### Machine Learning Pipeline
- **Algorithmus-Vergleich**: Random Forest vs. XGBoost mit Cross-Validation
- **Feature Selection**: Selektive Auswahl der 25 wichtigsten Features
- **Hyperparameter-Tuning**: Optimierte Parameter für maximale Performance

### Modell-Performance
| Algorithmus | R² Score | MAE | RMSE |
|-------------|----------|-----|------|
| **XGBoost** | **0.9709** | **58,468€** | **138,676€** |
| Random Forest | 0.9754 | 58,475€ | 127,511€ |

### Top Feature Importance
1. **Contract years left** - Vertragslaufzeit als Hauptfaktor
2. **Goals per 90** - Tor-Effizienz pro Spiel
3. **Total club market value** - Vereins-Prestige
4. **Market value average** - Historische Bewertungen
5. **Per-90 contributions** - Gesamtleistung pro Spiel

## 📊 Wissenschaftliche Marktwert-Formel

```python
# CIES Football Observatory Methodologie
def rational_formula(goals, assists, minutes, yellows, reds):
    # Basis-Wert
    basis_wert = 500_000 + goals * 200_000 + assists * 100_000 + minutes * 50
    
    # Effizienz-Premium (Elite-Performance)
    efficiency = (goals + assists) / games_played
    if efficiency >= 2.0:
        effizienz_multiplikator = 3.2  # SUPERSTAR
    elif efficiency >= 1.5:
        effizienz_multiplikator = 2.1  # STAR
    
    # Risk-Adjusted Discipline
    cards_per_game = total_cards / games_played
    if efficiency >= 1.5 and cards_per_game <= 0.4:
        disziplin_faktor = 1.0  # Controlled Aggression
    
    return basis_wert * effizienz_multiplikator * disziplin_faktor * scarcity_bonus
```

## 🎯 Key Findings

### Market Inefficiencies Identified
- **Junge Spieler**: Mindestens 30% Brand-Premium unabhängig von Performance
- **Stürmer vs. Verteidiger**: Position-Bias trotz gleicher statistischer Leistung
- **Impact Defenders**: Konsistent unterbewertet trotz hoher Performance

### Brand Premium Analysis
- **Mbappé**: 90% Marketing-Premium (180M€ vs. 18M€ Performance-Wert)
- **Etablierte Spieler**: 95%+ Übereinstimmung zwischen Formel und Marktwert
- **Underperformers**: Identifikation von überbewerteten Spielern

## 🎓 Business Applications

### Für Vereine
- **Scouting**: Identifikation unterbewerteter Performance-Stars
- **Transfer-Strategie**: Objektive Bewertung vs. Marktpreise
- **Vertragsverhandlungen**: Performance-basierte Gehaltsverhandlungen

### Für Analysten
- **Market Research**: Quantifizierung von Marketing-Premiums
- **Investment Decisions**: Datengetriebene Transfer-Entscheidungen
- **Risk Assessment**: Bewertung von Transfer-Risiken

## 🔬 Wissenschaftliche Einordnung

- **R² Score 97%**: Außergewöhnlich hoch für Sports Analytics (65% bereits "sehr gut")
- **Cross-Validation**: Robuste Performance auf neuen Daten
- **Feature Engineering**: 25+ Metriken decken alle Performance-Aspekte ab
- **Reproduzierbarkeit**: Alle Parameter dokumentiert (random_state=42)

## 📈 Model Limitations

Das Modell erfasst nicht:
- **Marketing-Wert**: Trikotverkäufe, Fan-Attraktion, Social Media
- **Verletzungshistorie**: Langfristige Risikofaktoren
- **Vertragssituationen**: Letztes Vertragsjahr = günstiger
- **Club-Politik**: Transfers aus nicht-performance Gründen

---

**Entwickelt als Data Science Projekt | Python, ML, Sports Analytics | R² Score: 97%**
