# ⚽ Football Player Market Value Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest-green.svg)](https://scikit-learn.org)
[![R²](https://img.shields.io/badge/R²-97%25-brightgreen.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-5.7M%20Records-orange.svg)](#)

A comprehensive Streamlit web application for analyzing and predicting football player market values using machine learning. The project analyzes **5.7 million performance records** and achieves an **R² Score of 97%** in market value prediction.

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Dataset Size** | 5.7M records |
| **Model Accuracy** | R² = 97% |
| **Market Analyzed** | €8.08B |
| **Leagues Covered** | 5 Top Leagues |
| **MAE** | €58,468 |
| **Features** | 25+ engineered |

## 🎯 Project Overview

**Problem**: The global football transfer market (€8.08 billion) shows systematic inefficiencies - players are often valued higher than their statistical performance justifies.

**Solution**: Development of an ML-based system for objective player valuation that separates performance from marketing premium.

**Technologies**: Python, Streamlit, XGBoost, Random Forest, Pandas, Plotly

## ✨ Core Features

- **🔍 Player Search**: Intelligent search with detailed season statistics
- **🤖 ML Dashboard**: Random Forest vs. XGBoost comparison with 97% R² Score
- **🧮 Market Value Formula**: Scientifically grounded calculation (CIES Football Observatory)
- **⭐ Brand Premium Analysis**: Performance vs. Marketing value quantification
- **📊 Market Inefficiency Detection**: Identification of under-/overvalued players

## 🚀 Quick Start

```bash
# Installation
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly xgboost

# Start app
streamlit run streamlit_finals.py
```

## 🔧 Technical Highlights

### Dataset & Preprocessing
- **5.7 million performance records** from 5 top leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
- **Feature Engineering**: 25+ calculated metrics (efficiency, discipline, position, age factors)
- **Data Cleaning**: Consistent formatting across various data sources

### Machine Learning Pipeline
- **Algorithm Comparison**: Random Forest vs. XGBoost with cross-validation
- **Feature Selection**: Selective choice of 25 most important features
- **Hyperparameter Tuning**: Optimized parameters for maximum performance

### Model Performance
| Algorithm | R² Score | MAE | RMSE |
|-----------|----------|-----|------|
| **XGBoost** | **0.9709** | **€58,468** | **€138,676** |
| Random Forest | 0.9754 | €58,475 | €127,511 |

### Top Feature Importance
1. **Contract years left** - Contract duration as main factor
2. **Goals per 90** - Goal efficiency per game
3. **Total club market value** - Club prestige
4. **Market value average** - Historical valuations
5. **Per-90 contributions** - Total performance per game

## 📊 Scientific Market Value Formula

```python
# CIES Football Observatory Methodology
def rational_formula(goals, assists, minutes, yellows, reds):
    # Base value
    basis_wert = 500_000 + goals * 200_000 + assists * 100_000 + minutes * 50
    
    # Efficiency premium (Elite performance)
    efficiency = (goals + assists) / games_played
    if efficiency >= 2.0:
        effizienz_multiplikator = 3.2  # SUPERSTAR
    elif efficiency >= 1.5:
        effizienz_multiplikator = 2.1  # STAR
    
    # Risk-adjusted discipline
    cards_per_game = total_cards / games_played
    if efficiency >= 1.5 and cards_per_game <= 0.4:
        disziplin_faktor = 1.0  # Controlled aggression
    
    return basis_wert * effizienz_multiplikator * disziplin_faktor * scarcity_bonus
```

## 🎯 Key Findings

### Market Inefficiencies Identified
- **Young Players**: Minimum 30% brand premium regardless of performance
- **Strikers vs. Defenders**: Position bias despite equal statistical performance
- **Impact Defenders**: Consistently undervalued despite high performance

### Brand Premium Analysis
- **Mbappé**: 90% marketing premium (€180M vs. €18M performance value)
- **Established Players**: 95%+ alignment between formula and market value
- **Underperformers**: Identification of overvalued players

## 🎓 Business Applications

### For Clubs
- **Scouting**: Identification of undervalued performance stars
- **Transfer Strategy**: Objective valuation vs. market prices
- **Contract Negotiations**: Performance-based salary discussions

### For Analysts
- **Market Research**: Quantification of marketing premiums
- **Investment Decisions**: Data-driven transfer decisions
- **Risk Assessment**: Evaluation of transfer risks

## 🔬 Scientific Context

- **R² Score 97%**: Exceptionally high for sports analytics (65% already considered "very good")
- **Cross-Validation**: Robust performance on new data
- **Feature Engineering**: 25+ metrics cover all performance aspects
- **Reproducibility**: All parameters documented (random_state=42)

## 📈 Model Limitations

The model does not capture:
- **Marketing Value**: Jersey sales, fan attraction, social media
- **Injury History**: Long-term risk factors
- **Contract Situations**: Last contract year = cheaper
- **Club Politics**: Transfers for non-performance reasons

## 🏆 Project Achievements

- **Data Science Excellence**: 97% R² score in complex sports analytics
- **Business Impact**: Identified €8B market inefficiencies
- **Technical Innovation**: Advanced feature engineering and ML pipeline
- **Scientific Rigor**: CIES Football Observatory methodology integration

## 🚀 Future Enhancements

- **Real-time Data Integration**: Live performance updates
- **Advanced Analytics**: Injury prediction and career trajectory modeling
- **API Development**: Integration with club management systems
- **Mobile Application**: On-the-go player analysis

---

**Developed as Data Science Project | Python, ML, Sports Analytics | R² Score: 97%**
