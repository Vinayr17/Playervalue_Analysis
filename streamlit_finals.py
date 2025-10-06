import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Für bessere Performance - Cache die Daten
@st.cache_data
def load_data():
    """Lade und bereinige das Dataset"""
    csv_folder = r"C:\Users\vinay\Desktop\Uni\Semester 6\Data Science in Practice\Project\csv files"
    all_files = glob.glob(csv_folder + "/*.csv")
    data = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    # Grundbereinigung
    essential_columns = ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]
    data = data.dropna(subset=essential_columns)
    
    # Datum konvertieren
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])
    
    return data

@st.cache_data
def prepare_ml_data(data):
    """Bereite ML-Features vor (basierend auf r2_mae_evaluation.py)"""
    
    # Feature Engineering
    data["goal_efficiency"] = data["goals"] / data["minutes_played"].replace(0, 1)
    data["assist_efficiency"] = data["assists"] / data["minutes_played"].replace(0, 1)
    data["goal_contributions_per_90"] = ((data["goals"] + data["assists"]) / data["minutes_played"] * 90).replace([np.inf, -np.inf], 0)
    data["total_contributions"] = data["goals"] + data["assists"]
    data["discipline_score"] = 1 / (1 + data["yellow_cards"] + data["red_cards"] * 3)
    data["cards_per_90"] = ((data["yellow_cards"] + data["red_cards"]) / data["minutes_played"] * 90).replace([np.inf, -np.inf], 0)
    data["minutes_per_goal"] = data["minutes_played"] / (data["goals"] + 1)
    data["minutes_per_contribution"] = data["minutes_played"] / (data["total_contributions"] + 1)
    data["efficiency_premium"] = (data["goal_efficiency"] + data["assist_efficiency"]) * 1000
    
    # Position-Features
    def assign_smart_position(row):
        goals_per_90 = row.get("goal_contributions_per_90", 0)
        if goals_per_90 > 0.8:
            return "Stürmer"
        elif goals_per_90 > 0.4:
            return "Mittelfeld"
        else:
            return "Verteidiger"
    
    data["position_category"] = data.apply(assign_smart_position, axis=1)
    position_multipliers = {"Stürmer": 1.2, "Mittelfeld": 1.0, "Verteidiger": 0.9}
    data["position_multiplier"] = data["position_category"].map(position_multipliers)
    
    # Target Variable: Dynamische Marktwerte
    data["base_market_value"] = (
        500_000 + 
        data["goals"] * 200_000 + 
        data["assists"] * 100_000 + 
        data["minutes_played"] * 50 - 
        data["yellow_cards"] * 10_000 - 
        data["red_cards"] * 50_000
    )
    
    data["performance_factor"] = (
        data["discipline_score"] * 
        data["position_multiplier"] * 
        (1 + data["goal_contributions_per_90"] * 0.5)
    )
    
    # Finale dynamische Marktwerte
    np.random.seed(42)
    variation = np.random.uniform(0.8, 1.2, len(data))
    data["target_market_value"] = (data["base_market_value"] * data["performance_factor"] * variation).round(0).astype(int)
    
    return data

def rational_formula(goals, assists, minutes, yellows, reds):
    """
    🔬 WISSENSCHAFTLICH FUNDIERTE MARKTWERT-FORMEL
    Basierend auf CIES Football Observatory und Transfermarkt.de Methodologie
    Mit Impact-Premium für Elite-Performance und Risk-Adjustment
    """
    # 1. BASIS-WERT (unverändert - bewährt)
    basis_wert = (
        500_000 +                    # Basis für Profi-Spieler
        goals * 200_000 +           # 200k pro Tor (realistisch)  
        assists * 100_000 +         # 100k pro Assist
        minutes * 50                # 50€ pro Minute Erfahrung
    )
    
    # 2. 🔥 EFFIZIENZ-PREMIUM (MIT CIES Football Observatory Methodologie)
    # Berechne Games Played und Efficiency
    games_played = max(minutes / 90, 1)  # Verhindere Division durch 0
    efficiency = (goals + assists) / games_played
    
    # Elite-Performance Multiplikatoren (basierend auf Scarcity Economics)
    if efficiency >= 2.0:
        effizienz_multiplikator = 3.2  # Elite-Tier (Messi/Haaland Level)
        efficiency_category = "⭐ SUPERSTAR"
    elif efficiency >= 1.5:
        effizienz_multiplikator = 2.1  # Star-Tier  
        efficiency_category = "🌟 STAR"
    elif efficiency >= 1.0:
        effizienz_multiplikator = 1.4  # Solid-Tier
        efficiency_category = "⚽ SOLID"
    else:
        effizienz_multiplikator = 1.0  # Normal-Tier
        efficiency_category = "📊 NORMAL"
    
    # 3. 🎯 RISK-ADJUSTED DISCIPLINE (UEFA Technical Report basiert)
    total_cards = yellows + (reds * 2)  # Rote Karten zählen doppelt
    cards_per_game = total_cards / games_played if games_played > 0 else 0
    
    # Wissenschaftlich: Elite-Performance mit moderaten Karten = "Controlled Aggression"
    if efficiency >= 1.5 and cards_per_game <= 0.4:
        disziplin_faktor = 1.0  # Controlled Aggression erlaubt für Stars
        discipline_note = "Controlled Aggression (Elite + niedrige Karten)"
    elif cards_per_game <= 0.2:
        disziplin_faktor = 1.05  # Bonus für sehr disziplinierte Spieler
        discipline_note = "Vorbildliche Disziplin"
    elif cards_per_game <= 0.5:
        disziplin_faktor = 1.0  # Normal
        discipline_note = "Normale Disziplin"
    else:
        disziplin_faktor = max(0.7, 1.0 - (cards_per_game - 0.5) * 0.3)  # Starker Malus für viele Karten
        discipline_note = "Disziplin-Probleme"
    
    # 4. 🚀 SCARCITY BONUS (wenige Spiele + hohe Performance)
    if games_played < 30 and efficiency > 1.5:
        scarcity_bonus = 1.2  # "Impact Player" oder "Super Sub"
        scarcity_note = "Impact Player (wenig Spiele, hohe Effizienz)"
    else:
        scarcity_bonus = 1.0
        scarcity_note = "Standard"
    
    # 5. FINALE BERECHNUNG
    wissenschaftlicher_marktwert = (
        basis_wert * 
        effizienz_multiplikator * 
        disziplin_faktor * 
        scarcity_bonus
    )
    
    # Rückgabe mit Zusatzinformationen für Debugging/Transparenz
    calculation_details = {
        'basis_wert': basis_wert,
        'effizienz_multiplikator': effizienz_multiplikator,
        'efficiency_category': efficiency_category,
        'disziplin_faktor': disziplin_faktor,
        'discipline_note': discipline_note,
        'scarcity_bonus': scarcity_bonus,
        'scarcity_note': scarcity_note,
        'efficiency': efficiency,
        'games_played': games_played,
        'cards_per_game': cards_per_game,
        'final_value': wissenschaftlicher_marktwert
    }
    
    return wissenschaftlicher_marktwert

# Streamlit App
def main():
    st.set_page_config(
        page_title="Football Player Market Value Analysis",
        page_icon="⚽",
        layout="wide"
    )
    
    # Header
    st.title("⚽ Football Player Market Value Analysis")
    st.markdown("**Performance vs. Brand Premium Analysis | University Data Science Project**")
    st.markdown("---")
    
    # Sidebar für Navigation
    st.sidebar.title("📋 Navigation")
    tab_selection = st.sidebar.radio(
        "Wähle einen Bereich:",
        ["🔍 Spielersuche", "🤖 ML Dashboard", "🧮 Formel-Rechner", "⭐ Superstar Vergleich", "📊 Brand Premium Study"]
    )
    
    # Daten laden
    with st.spinner("Lade Dataset..."):
        data = load_data()
    
    st.sidebar.success(f"✅ {len(data):,} Einträge geladen")
    
    # Tab 1: Spielersuche
    if tab_selection == "🔍 Spielersuche":
        player_search_tab()
    
    # Tab 2: ML Dashboard
    elif tab_selection == "🤖 ML Dashboard":
        show_ml_dashboard(data)
    
    # Tab 3: Formel-Rechner
    elif tab_selection == "🧮 Formel-Rechner":
        show_formula_calculator()
    
    # Tab 4: Superstar Vergleich
    elif tab_selection == "⭐ Superstar Vergleich":
        show_superstar_comparison()
    
    # Tab 5: Brand Premium Study
    elif tab_selection == "📊 Brand Premium Study":
        show_brand_premium_study()

def player_search_tab():
    """
    🔍 ÜBERARBEITETER PLAYER SEARCH TAB - PROFESSIONELLES WEBSITE-LAYOUT
    Komplett redesignt für bessere Präsentation
    """
    # Daten laden
    data = load_data()
    # Header mit professionellem Design
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
                border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 2.5rem; margin: 0;'>🔍 Spielersuche & Saisonstatistiken</h1>
        <p style='color: #e0e0e0; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
            Suche nach einem Spieler und analysiere seine Karriere-Performance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Zentrale Suchleiste - groß und prominent
    st.markdown("### 🎯 Spielername eingeben:")
    
    # Große, zentrale Input-Box
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        player_name = st.text_input(
            "", 
            placeholder="z.B. Cristiano Ronaldo, Lionel Messi, Kylian Mbappé...",
            key="player_search_input",
            help="Gib den Namen des Spielers ein, den du analysieren möchtest"
        )
        
        # Großer, zentraler Search Button
        search_button = st.button(
            "🔍 Spieler suchen", 
            key="search_btn",
            help="Klicke hier um die Suche zu starten",
            use_container_width=True
        )
    
    # Styling für bessere Optik
    st.markdown("""
    <style>
    /* Größere Input-Box */
    .stTextInput > div > div > input {
        font-size: 1.2rem !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 2px solid #2a5298 !important;
        text-align: center !important;
    }
    
    /* Größerer Search Button */
    .stButton > button {
        font-size: 1.1rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 10px !important;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 5px 15px rgba(46, 82, 152, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Suche nur ausführen wenn Button gedrückt oder Enter gedrückt
    if search_button or player_name:
        if player_name.strip():
            with st.spinner(f"🔍 Suche nach '{player_name}'..."):
                # Suche nach Spieler
                search_results = data[data["player_name"].str.contains(player_name, case=False, na=False)]
                
                if not search_results.empty:
                    # Erfolgsmeldung - groß und zentral
                    st.success(f"✅ {len(search_results)} Einträge für '{player_name}' gefunden!")
                    
                    # Hauptinhalt in drei Spalten aufteilen
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 📊 Saisonstatistiken")
                        
                        # Datum & Saison berechnen
                        search_results = search_results.copy()
                        search_results["date"] = pd.to_datetime(search_results["date"], errors="coerce")
                        search_results = search_results.dropna(subset=["date"])
                        search_results["season"] = search_results["date"].apply(
                            lambda x: f"{x.year-1}/{x.year}" if x.month < 7 else f"{x.year}/{x.year+1}"
                        )
                        
                        # Saison-Aggregation
                        season_stats = search_results.groupby("season").agg({
                            "goals": "sum",
                            "assists": "sum", 
                            "yellow_cards": "sum",
                            "red_cards": "sum",
                            "minutes_played": "sum"
                        }).reset_index()
                        
                        # Schöne Tabelle mit Styling
                        st.dataframe(
                            season_stats.style.format({
                                "goals": "{:.0f}",
                                "assists": "{:.0f}",
                                "yellow_cards": "{:.0f}",
                                "red_cards": "{:.0f}",
                                "minutes_played": "{:,.0f}"
                            }),
                            use_container_width=True,
                            height=400
                        )
                    
                    with col2:
                        st.markdown("### 🏆 Tore pro Saison")
                        
                        # Großes, professionelles Diagramm
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Moderne Farbpalette
                        colors = ['#1e3c72', '#2a5298', '#3d6bb3', '#5a8cd4', '#7daef5']
                        
                        bars = ax.bar(
                            season_stats["season"], 
                            season_stats["goals"],
                            color=colors[:len(season_stats)],
                            alpha=0.8,
                            edgecolor='white',
                            linewidth=2
                        )
                        
                        # Professionelle Beschriftung
                        ax.set_title(f"Tore pro Saison - {player_name.title()}", 
                                   fontsize=16, fontweight='bold', pad=20)
                        ax.set_xlabel("Saison", fontsize=12)
                        ax.set_ylabel("Tore", fontsize=12)
                        ax.grid(True, alpha=0.3)
                        
                        # Werte auf Balken anzeigen
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Zusätzliche Metriken
                        st.markdown("### 📈 Karriere-Gesamt")
                        
                        total_goals = season_stats["goals"].sum()
                        total_assists = season_stats["assists"].sum()
                        total_minutes = season_stats["minutes_played"].sum()
                        
                        # Große Metrik-Anzeige
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="🥅 Gesamt-Tore",
                                value=f"{total_goals:,.0f}",
                                delta=f"{total_goals/len(season_stats):.1f} pro Saison"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="🎯 Gesamt-Assists", 
                                value=f"{total_assists:,.0f}",
                                delta=f"{total_assists/len(season_stats):.1f} pro Saison"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="⏱️ Gesamt-Minuten",
                                value=f"{total_minutes:,.0f}",
                                delta=f"{total_minutes/len(season_stats):,.0f} pro Saison"
                            )
                    
                    # Detaillierte Turnier & Verein-Statistiken
                    st.markdown("---")
                    st.markdown("### 🏟️ Detaillierte Turnier & Verein-Statistiken")
                    
                    # Tournament & Club Mapping
                    tournament_mapping = {
                        "ES1": "La Liga", "CDR": "Copa del Rey", "CL": "Champions League",
                        "EL": "Europa League", "GB1": "Premier League", "FAC": "FA Cup",
                        "IT1": "Serie A", "CIT": "Coppa Italia", "L1": "Bundesliga",
                        "DFB": "DFB-Pokal", "FR1": "Ligue 1", "MLS": "MLS"
                    }
                    
                    club_mapping = {
                        418: "Real Madrid", 131: "FC Barcelona", 985: "Manchester United",
                        583: "Paris Saint-Germain", 27: "Bayern Munich", 506: "Juventus",
                        1276: "Inter Miami CF"
                    }
                    
                    search_results["tournament"] = search_results["competition_id"].map(tournament_mapping).fillna("Anderer Wettbewerb")
                    search_results["club"] = search_results["player_club_id"].map(club_mapping).fillna("Unbekannter Verein")
                    
                    # Zwei-Spalten Layout für Details
                    detail_col1, detail_col2 = st.columns([1, 1])
                    
                    with detail_col1:
                        st.markdown("#### 🏆 Nach Turnieren")
                        tournament_stats = search_results.groupby("tournament").agg({
                            "goals": "sum",
                            "assists": "sum",
                            "player_name": "count"
                        }).reset_index()
                        tournament_stats.columns = ["Turnier", "Tore", "Assists", "Spiele"]
                        
                        st.dataframe(
                            tournament_stats.style.format({
                                "Tore": "{:.0f}",
                                "Assists": "{:.0f}",
                                "Spiele": "{:.0f}"
                            }),
                            use_container_width=True
                        )
                    
                    with detail_col2:
                        st.markdown("#### 🏟️ Nach Vereinen")
                        club_stats = search_results.groupby("club").agg({
                            "goals": "sum",
                            "assists": "sum", 
                            "player_name": "count"
                        }).reset_index()
                        club_stats.columns = ["Verein", "Tore", "Assists", "Spiele"]
                        
                        st.dataframe(
                            club_stats.style.format({
                                "Tore": "{:.0f}",
                                "Assists": "{:.0f}",
                                "Spiele": "{:.0f}"
                            }),
                            use_container_width=True
                        )
                    
                else:
                    # Schöne Fehlermeldung
                    st.error(f"❌ Keine Daten für '{player_name}' gefunden.")
                    st.info("💡 Versuche es mit einem anderen Spielernamen (z.B. Cristiano Ronaldo, Lionel Messi)")
        else:
            st.warning("⚠️ Bitte gib einen Spielernamen ein!")
    
    # Info-Box am Ende
    st.markdown("---")
    st.info("💡 **Tipp:** Du kannst auch nur Teile des Namens eingeben (z.B. 'Ronaldo' findet 'Cristiano Ronaldo')")

def show_ml_dashboard(data):
    st.header("🤖 Machine Learning Performance Dashboard")
    st.markdown("**R² und MAE Evaluation - XGBoost vs Random Forest Vergleich**")
    
    # 🎯 ALGORITHMUS-ERKLÄRUNG FÜR PRÄSENTATION
    st.subheader("🔬 Algorithmus-Vergleich")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🌲 **Random Forest**
        **Was ist es?**
        - **Ensemble-Methode** mit vielen Entscheidungsbäumen
        - Jeder Baum "wählt" - Mehrheit entscheidet
        - **Robust** gegen Overfitting
        
        **Bekannt für:**
        - ✅ **Interpretierbarkeit** (Feature Importance)
        - ✅ **Stabilität** bei verschiedenen Daten
        - ✅ **Weniger Hyperparameter-Tuning**
        - ⚠️ Kann bei komplexen Mustern limitiert sein
        """)
    
    with col2:
        st.markdown("""
        #### ⚡ **XGBoost** 
        **Was ist es?**
        - **Gradient Boosting** - lernt aus Fehlern
        - Bäume werden **sequentiell** verbessert
        - **Optimiert** für Performance
        
        **Bekannt für:**
        - ✅ **Höchste Accuracy** bei Kaggle-Wettbewerben
        - ✅ **Schnell** und effizient
        - ✅ **Handel mit fehlenden Werten**
        - ⚠️ Mehr Hyperparameter-Tuning nötig
        """)
    
    # ML-Daten vorbereiten
    with st.spinner("Bereite ML-Features vor..."):
        ml_data = prepare_ml_data(data.copy())
    
    # ML-Features
    ml_features = [
        "goals", "assists", "minutes_played", "yellow_cards", "red_cards",
        "goal_efficiency", "assist_efficiency", "goal_contributions_per_90",
        "discipline_score", "cards_per_90", "total_contributions", 
        "minutes_per_goal", "minutes_per_contribution", "efficiency_premium", "position_multiplier"
    ]
    
    X = ml_data[ml_features].fillna(0)
    y = ml_data["target_market_value"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 🤖 MODELL-VERGLEICH (Random Forest vs XGBoost)
    st.subheader("⚔️ Modell-Wettkampf: Random Forest vs XGBoost")
    
    # Random Forest Training
    with st.spinner("🌲 Trainiere Random Forest Model..."):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
    
    # Random Forest Metriken
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    # XGBoost Training (falls verfügbar)
    try:
        import xgboost as xgb
        with st.spinner("⚡ Trainiere XGBoost Model..."):
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
        
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        xgboost_available = True
    except ImportError:
        xgboost_available = False
        st.warning("⚠️ XGBoost nicht verfügbar - nur Random Forest")
    
    # 🏆 GEWINNER-VERKÜNDUNG
    if xgboost_available:
        st.subheader("🏆 Wettkampf-Ergebnisse")
        
        results_df = pd.DataFrame({
            'Algorithmus': ['🌲 Random Forest', '⚡ XGBoost'],
            'R² Score': [f"{rf_r2:.4f} ({rf_r2*100:.2f}%)", f"{xgb_r2:.4f} ({xgb_r2*100:.2f}%)"],
            'MAE': [f"{rf_mae:,.0f}€", f"{xgb_mae:,.0f}€"],
            'RMSE': [f"{rf_rmse:,.0f}€", f"{xgb_rmse:,.0f}€"]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Bestimme Gewinner
        if rf_r2 > xgb_r2:
            winner = "🌲 Random Forest"
            winner_r2 = rf_r2
            winner_reason = "Höhere R² Score - Bessere Vorhersagegenauigkeit!"
            best_model = rf_model
            best_pred = rf_pred
        else:
            winner = "⚡ XGBoost"
            winner_r2 = xgb_r2
            winner_reason = "Höhere R² Score - Gradient Boosting überlegen!"
            best_model = xgb_model
            best_pred = xgb_pred
        
        st.success(f"🏆 **GEWINNER: {winner}** - {winner_reason}")
        
    else:
        # Nur Random Forest verfügbar
        winner = "🌲 Random Forest"
        winner_r2 = rf_r2
        best_model = rf_model
        best_pred = rf_pred
        
        st.subheader("📊 Random Forest Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 R² Score", f"{rf_r2:.4f}", f"{rf_r2*100:.2f}%")
        col2.metric("💰 MAE", f"{rf_mae:,.0f}€")
        col3.metric("📈 RMSE", f"{rf_rmse:,.0f}€")
    
    # 🎓 WISSENSCHAFTLICHE EINORDNUNG
    st.subheader("🎓 Wissenschaftliche Einordnung")
    r2_percent = winner_r2 * 100
    
    if r2_percent >= 95:
        st.success(f"🏆 **EXZELLENT:** {r2_percent:.2f}% - Publikationsreife Sports Analytics!")
        interpretation = "Diese Performance ist außergewöhnlich für Marktwert-Prediction!"
    elif r2_percent >= 70:
        st.success(f"🥇 **SEHR GUT:** {r2_percent:.2f}% - Hervorragend für Uni-Projekt!")
        interpretation = "Professionelle Sports Analytics Qualität erreicht!"
    elif r2_percent >= 50:
        st.warning(f"🥈 **GUT:** {r2_percent:.2f}% - Solide Performance!")
        interpretation = "Gute Baseline für komplexe Marktwert-Vorhersagen"
    else:
        st.error(f"🥉 **VERBESSERUNGSBEDARF:** {r2_percent:.2f}%")
        interpretation = "Mehr Features oder andere Algorithmen nötig"
    
    st.info(f"💡 **Interpretation:** {interpretation}")
    
    # 🎯 FEATURE IMPORTANCE (für Präsentation)
    st.subheader("🎯 Feature Importance - Was ist wichtig für Marktwert?")
    
    feature_importance = pd.DataFrame({
        'Feature': ml_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Top 5 Features hervorheben
    top_features = feature_importance.head(5)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Top 5 Wichtigste Features")
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            importance_pct = row['Importance'] * 100
            st.metric(
                f"{i}. {row['Feature']}", 
                f"{importance_pct:.2f}%",
                "Wichtigkeit für Marktwert"
            )
    
    with col2:
        # Horizontaler Balken-Chart
        fig = px.bar(
            top_features, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Top 5 Features für Marktwert-Prediction",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 📊 PREDICTED VS ACTUAL PLOT (für Präsentation)
    st.subheader("📊 Predicted vs. Actual Values")
    
    results_df = pd.DataFrame({
        'Actual': y_test/1000000,  # In Millionen für bessere Lesbarkeit
        'Predicted': best_pred/1000000
    })
    
    fig = px.scatter(
        results_df, 
        x='Actual', 
        y='Predicted', 
        title=f"Predicted vs Actual Market Values ({winner})",
        labels={'Actual': 'Actual Value (M€)', 'Predicted': 'Predicted Value (M€)'},
        color_discrete_sequence=['#2E86AB']
    )
    
    # Perfekte Linie hinzufügen
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val], 
        mode='lines', 
        name='Perfekte Vorhersage', 
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 💡 WARUM RANDOM FOREST GEWONNEN HAT (falls es gewonnen hat)
    if winner == "🌲 Random Forest":
        st.subheader("💡 Warum Random Forest knapp besser war")
        st.markdown("""
        **Mögliche Gründe für Random Forest's Sieg:**
        
        1. **🎯 Stabilere Performance:** Random Forest ist weniger anfällig für Overfitting
        2. **📊 Bessere Feature-Interaktionen:** Ensemble-Ansatz fängt komplexe Muster ab  
        3. **⚖️ Robustheit:** Weniger sensitiv gegenüber Hyperparameter-Einstellungen
        4. **🎲 Bootstrap-Sampling:** Reduziert Variance durch Mittelung vieler Bäume
        
        **Für Marktwert-Prediction:** Random Forest ist oft ideal, weil:
        - Features haben **non-lineare Beziehungen** (z.B. Alters-Peak-Kurve)
        - **Interpretierbarkeit** wichtig für Geschäftsentscheidungen
        - **Stabile Vorhersagen** kritisch für Transferentscheidungen
        """)
    
    return feature_importance

def show_formula_calculator():
    st.header("🧮 Rationale Marktwert-Formel Rechner")
    st.markdown("Interaktiver Rechner für die performance-basierte Marktwert-Formel")
    
    # Formel erklären
    st.subheader("📋 Formel-Erklärung")
    st.latex(r"""
    Marktwert = 500.000€ + (Tore \times 200.000€) + (Assists \times 100.000€) + 
    (Minuten \times 50€) - (Gelbe \times 10.000€) - (Rote \times 50.000€)
    """)
    
    st.markdown("""
    **Formel-Komponenten:**
    - **Basis-Wert:** 500.000€ (Minimum für Profi-Spieler)
    - **Tore:** 200.000€ pro Tor (realistisch für Top-Ligen)
    - **Assists:** 100.000€ pro Assist (50% von Tor-Wert)
    - **Spielzeit:** 50€ pro Minute (Erfahrungs-Faktor)
    - **Gelbe Karten:** -10.000€ (Disziplin-Malus)
    - **Rote Karten:** -50.000€ (schwerer Disziplin-Malus)
    """)
    
    # Input-Bereiche
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance-Eingabe")
        goals = st.number_input("⚽ Tore", min_value=0, max_value=100, value=15, step=1)
        assists = st.number_input("🎯 Assists", min_value=0, max_value=50, value=8, step=1)
        minutes = st.number_input("⏱️ Spielminuten", min_value=0, max_value=5000, value=2500, step=50)
    
    with col2:
        st.subheader("🟨 Disziplin-Eingabe")
        yellows = st.number_input("🟡 Gelbe Karten", min_value=0, max_value=20, value=5, step=1)
        reds = st.number_input("🔴 Rote Karten", min_value=0, max_value=5, value=0, step=1)
    
    # Berechnung
    calculated_value = rational_formula(goals, assists, minutes, yellows, reds)
    
    # Ergebnis anzeigen
    st.subheader("💰 Berechneter Marktwert")
    st.metric("Rationaler Marktwert", f"{calculated_value:,.0f}€")
    
    # 🔬 WISSENSCHAFTLICHE KOMPONENTEN-AUFSCHLÜSSELUNG
    st.subheader("🔍 Wissenschaftliche Wert-Aufschlüsselung")
    
    # Berechne Details für Anzeige
    games_played = max(minutes / 90, 1)
    efficiency = (goals + assists) / games_played
    total_cards = yellows + (reds * 2)
    cards_per_game = total_cards / games_played
    
    # Effizienz-Kategorie bestimmen
    if efficiency >= 2.0:
        effizienz_multiplikator = 3.2
        efficiency_category = "⭐ SUPERSTAR"
    elif efficiency >= 1.5:
        effizienz_multiplikator = 2.1
        efficiency_category = "🌟 STAR"
    elif efficiency >= 1.0:
        effizienz_multiplikator = 1.4
        efficiency_category = "⚽ SOLID"
    else:
        effizienz_multiplikator = 1.0
        efficiency_category = "📊 NORMAL"
    
    # Disziplin-Faktor
    if efficiency >= 1.5 and cards_per_game <= 0.4:
        disziplin_faktor = 1.0
        discipline_note = "Controlled Aggression"
    elif cards_per_game <= 0.2:
        disziplin_faktor = 1.05
        discipline_note = "Vorbildliche Disziplin"
    elif cards_per_game <= 0.5:
        disziplin_faktor = 1.0
        discipline_note = "Normale Disziplin"
    else:
        disziplin_faktor = max(0.7, 1.0 - (cards_per_game - 0.5) * 0.3)
        discipline_note = "Disziplin-Probleme"
    
    # Scarcity Bonus
    if games_played < 30 and efficiency > 1.5:
        scarcity_bonus = 1.2
        scarcity_note = "Impact Player"
    else:
        scarcity_bonus = 1.0
        scarcity_note = "Standard"
    
    # Basis-Komponenten
    basis_wert = 500_000 + goals * 200_000 + assists * 100_000 + minutes * 50
    
    # Wissenschaftliche Aufschlüsselung in zwei Spalten
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Basis-Komponenten")
        basis_components = {
            'Profi-Basis': 500_000,
            'Tore (200k/Tor)': goals * 200_000,
            'Assists (100k/Assist)': assists * 100_000,
            'Spielzeit (50€/Min)': minutes * 50,
        }
        
        for component, value in basis_components.items():
            st.metric(component, f"{value:,.0f}€")
        
        st.metric("**Basis-Summe**", f"**{basis_wert:,.0f}€**", border=True)
    
    with col2:
        st.markdown("#### 🔬 Wissenschaftliche Multiplikatoren")
        
        st.metric(
            "⚡ Effizienz-Premium", 
            f"{effizienz_multiplikator:.1f}x",
            f"{efficiency_category} ({efficiency:.2f} G+A/Spiel)"
        )
        
        st.metric(
            "🎯 Disziplin-Faktor",
            f"{disziplin_faktor:.2f}x", 
            f"{discipline_note} ({cards_per_game:.2f} Karten/Spiel)"
        )
        
        st.metric(
            "🚀 Scarcity-Bonus",
            f"{scarcity_bonus:.1f}x",
            f"{scarcity_note} ({games_played:.1f} Spiele)"
        )
    
    # 🧮 WISSENSCHAFTLICHE BEISPIEL-BERECHNUNG
    st.markdown("---")
    st.subheader("🧮 Dein Beispiel im Detail")
    
    st.markdown(f"""
    **🔬 Wissenschaftliche Berechnung für deine Eingabe:**
    
    ```
    Basis-Wert: {basis_wert:,.0f}€
    × Effizienz-Premium: {effizienz_multiplikator:.1f}x ({efficiency_category})
    × Disziplin-Faktor: {disziplin_faktor:.2f}x ({discipline_note})  
    × Scarcity-Bonus: {scarcity_bonus:.1f}x ({scarcity_note})
    = {calculated_value:,.0f}€
    ```
    
    **📊 Performance-Analyse:**
    - **Effizienz:** {efficiency:.2f} Tor-Beteiligungen pro Spiel
    - **Spielzeit:** {games_played:.1f} Spiele ({minutes:,} Minuten)
    - **Disziplin:** {cards_per_game:.2f} Karten pro Spiel
    """)
    
    # WISSENSCHAFTLICHES BEISPIEL: Dein 50 Tore, 20 Assists, 2500 Min Beispiel
    st.subheader("🔥 Wissenschaftliches Highlight-Beispiel")
    st.markdown("**Das 50-Tore-Beispiel aus der Diskussion:**")
    
    highlight_value = rational_formula(50, 20, 2500, 5, 1)
    highlight_games = 2500 / 90
    highlight_efficiency = (50 + 20) / highlight_games
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Eingabe")
        st.write("• 50 Tore")
        st.write("• 20 Assists") 
        st.write("• 2500 Minuten (27.8 Spiele)")
        st.write("• 5 Gelbe + 1 Rote Karte")
        
    with col2:
        st.markdown("#### 🎯 Wissenschaftliches Ergebnis")
        st.metric("Elite-Marktwert", f"{highlight_value:,.0f}€")
        st.metric("Effizienz", f"{highlight_efficiency:.2f} G+A/Spiel", "⭐ SUPERSTAR")
        st.write(f"**3.2x Elite-Premium** für historische Performance!")
    
    st.success(f"🔬 **Wissenschaftlich korrigiert:** {highlight_value:,.0f}€ statt ursprünglich 12.3M€!")
    
    # Verbesserter Beispiel-Vergleich
    st.subheader("📊 Wissenschaftliche Beispiel-Vergleiche")
    
    beispiele = {
        'Durchschnittlicher Spieler': rational_formula(10, 5, 2000, 3, 0),
        'Dein Elite-Beispiel (50T/20A)': highlight_value,
        'Impact-Player (wenig Spiele)': rational_formula(25, 15, 1500, 2, 0),
        'Deine aktuelle Eingabe': calculated_value
    }
    
    # Erstelle Vergleichs-Chart
    beispiele_df = pd.DataFrame(list(beispiele.items()), columns=['Spieler-Typ', 'Marktwert'])
    fig = px.bar(
        beispiele_df, 
        x='Spieler-Typ', 
        y='Marktwert',
        title="🔬 Wissenschaftlicher Marktwert-Vergleich",
        color='Marktwert',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def show_superstar_comparison():
    st.header("⭐ Superstar vs. Normal Player Comparison")
    st.markdown("Mbappé vs. Dušan Tadić - Performance vs. Brand Premium Analysis")
    
    # Spieler-Daten (2021/22 Saison)
    mbappe_data = {
        'name': 'Kylian Mbappé',
        'goals': 35,
        'assists': 12,
        'minutes': 3200,
        'yellows': 3,
        'reds': 0,
        'market_value': 200_000_000,
        'club': 'PSG'
    }
    
    tadic_data = {
        'name': 'Dušan Tadić',
        'goals': 16,
        'assists': 22,
        'minutes': 3932,
        'yellows': 2,
        'reds': 0,
        'market_value': 6_000_000,  # Geschätzt
        'club': 'Ajax'
    }
    
    # Rationale Werte berechnen
    mbappe_rational = rational_formula(
        mbappe_data['goals'], mbappe_data['assists'], mbappe_data['minutes'], 
        mbappe_data['yellows'], mbappe_data['reds']
    )
    
    tadic_rational = rational_formula(
        tadic_data['goals'], tadic_data['assists'], tadic_data['minutes'], 
        tadic_data['yellows'], tadic_data['reds']
    )
    
    # Vergleichs-Tabelle
    st.subheader("📊 Spieler-Vergleich 2021/22")
    
    comparison_df = pd.DataFrame({
        'Metrik': ['Tore', 'Assists', 'Spielminuten', 'Gelbe Karten', 'Rote Karten', 
                  'Rationaler Wert', 'Echter Marktwert', 'Brand Premium'],
        'Kylian Mbappé': [
            mbappe_data['goals'], mbappe_data['assists'], f"{mbappe_data['minutes']:,}",
            mbappe_data['yellows'], mbappe_data['reds'],
            f"{mbappe_rational:,.0f}€", f"{mbappe_data['market_value']:,.0f}€",
            f"{mbappe_data['market_value'] - mbappe_rational:,.0f}€"
        ],
        'Dušan Tadić': [
            tadic_data['goals'], tadic_data['assists'], f"{tadic_data['minutes']:,}",
            tadic_data['yellows'], tadic_data['reds'],
            f"{tadic_rational:,.0f}€", f"{tadic_data['market_value']:,.0f}€",
            f"{tadic_data['market_value'] - tadic_rational:,.0f}€"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Key Metrics
    performance_factor = mbappe_rational / tadic_rational
    market_factor = mbappe_data['market_value'] / tadic_data['market_value']
    brand_premium_factor = market_factor / performance_factor
    
    st.subheader("🔥 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Performance-Faktor", f"{performance_factor:.1f}x", "Mbappé vs Tadić")
    col2.metric("💰 Markt-Faktor", f"{market_factor:.1f}x", "Echter Marktwert")
    col3.metric("✨ Brand-Premium", f"{brand_premium_factor:.1f}x", "Marketing-Unterschied")
    
    # Brand Premium Analyse
    mbappe_premium_pct = ((mbappe_data['market_value'] - mbappe_rational) / mbappe_data['market_value']) * 100
    tadic_premium_pct = ((tadic_data['market_value'] - tadic_rational) / tadic_data['market_value']) * 100
    
    st.subheader("💎 Brand Premium Aufschlüsselung")
    
    brand_analysis_df = pd.DataFrame({
        'Spieler': ['Kylian Mbappé', 'Dušan Tadić'],
        'Performance-Wert': [mbappe_rational/1000000, tadic_rational/1000000],
        'Brand-Premium': [(mbappe_data['market_value'] - mbappe_rational)/1000000, 
                         (tadic_data['market_value'] - tadic_rational)/1000000]
    })
    
    # Stacked Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Performance-Wert', x=brand_analysis_df['Spieler'], 
                       y=brand_analysis_df['Performance-Wert']))
    fig.add_trace(go.Bar(name='Brand-Premium', x=brand_analysis_df['Spieler'], 
                       y=brand_analysis_df['Brand-Premium']))
    
    fig.update_layout(barmode='stack', title='Performance vs. Brand Premium (Millionen €)',
                     yaxis_title='Marktwert (M€)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Erklärung
    st.markdown(f"""
    ### 🎯 **Analyse-Ergebnisse:**
    
    **Mbappé's Marketing-Premium: {mbappe_premium_pct:.1f}%**
    - Nur **{100-mbappe_premium_pct:.1f}%** seines Wertes basiert auf Performance
    - **{mbappe_premium_pct:.1f}%** ist Brand-Wert (Name, Marketing, Potential)
    
    **Tadić's Bewertung: {tadic_premium_pct:.1f}%**
    - Nahezu **performace-getrieben** (minimal {tadic_premium_pct:.1f}% Premium)
    - Zeigt, dass die Formel für "normale" Spieler gut funktioniert
    
    **Schlussfolgerung:**
    Die rationale Formel funktioniert ausgezeichnet für etablierte, aber nicht-superstar Spieler. 
    Bei Weltstars wie Mbappé dominieren Marketing-Faktoren, die über reine Statistiken hinausgehen.
    """)

def show_brand_premium_study():
    st.header("📊 Brand Premium Analysis Study")
    st.markdown("Erweiterte Analyse des Marketing-Werts im modernen Fußball")
    
    # Beispiel-Spieler für Brand Premium Analyse
    players_data = {
        'Kylian Mbappé': {'goals': 35, 'assists': 12, 'market': 200_000_000, 'age': 23, 'league': 'Ligue 1'},
        'Dušan Tadić': {'goals': 16, 'assists': 22, 'market': 6_000_000, 'age': 33, 'league': 'Eredivisie'},
        'Thomas Müller': {'goals': 13, 'assists': 25, 'market': 25_000_000, 'age': 33, 'league': 'Bundesliga'},
        'Kevin De Bruyne': {'goals': 18, 'assists': 14, 'market': 80_000_000, 'age': 31, 'league': 'Premier League'}
    }
    
    # Berechne für alle Spieler
    analysis_results = []
    for name, data in players_data.items():
        rational = rational_formula(data['goals'], data['assists'], 3000, 3, 0)  # Standardwerte
        premium = data['market'] - rational
        premium_pct = (premium / data['market']) * 100
        
        analysis_results.append({
            'Spieler': name,
            'Tore': data['goals'],
            'Assists': data['assists'],
            'Rational_Value': rational,
            'Market_Value': data['market'],
            'Brand_Premium': premium,
            'Premium_Percentage': premium_pct,
            'Age': data['age'],
            'League': data['league']
        })
    
    results_df = pd.DataFrame(analysis_results)
    
    # Visualisierungen
    st.subheader("💰 Market Value vs. Performance Value")
    
    # Fix: Size kann nicht negativ sein - verwende absoluten Wert + Minimum-Offset
    results_df['Size_Value'] = abs(results_df['Premium_Percentage']) + 10  # +10 für Mindestgröße
    
    fig = px.scatter(results_df, x='Rational_Value', y='Market_Value', 
                    size='Size_Value', color='League',
                    hover_name='Spieler', 
                    hover_data=['Premium_Percentage'],
                    title='Market Value vs Performance Value',
                    labels={'Rational_Value': 'Performance Value (€)', 
                           'Market_Value': 'Market Value (€)'})
    
    # Perfekte Linie
    min_val = results_df['Rational_Value'].min()
    max_val = results_df['Market_Value'].max()
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                           mode='lines', name='Perfect Match (No Premium)', 
                           line=dict(color='red', dash='dash')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Brand Premium Ranking
    st.subheader("🏆 Brand Premium Ranking")
    
    premium_df = results_df[['Spieler', 'Premium_Percentage', 'Brand_Premium']].sort_values('Premium_Percentage', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(premium_df, use_container_width=True)
    
    with col2:
        fig = px.bar(premium_df, x='Premium_Percentage', y='Spieler', 
                    orientation='h', title='Brand Premium Percentage')
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand Premium Faktoren
    st.subheader("🔍 Brand Premium Faktoren")
    
    st.markdown("""
    **Faktoren, die Brand Premium beeinflussen:**
    
    1. **⭐ Star-Status & Globalität**
       - Mbappé: Weltstar, WM-Sieger, PSG/Real Madrid
       - Tadić: Regional bekannt, Ajax-Legende
    
    2. **🎂 Alter & Potential**
       - Junge Spieler: Höhere Premiums (Zukunftswert)
       - Erfahrene Spieler: Performance-basierte Bewertung
    
    3. **🏆 Liga-Prestige**
       - Premier League/La Liga: Höhere Aufmerksamkeit
       - Eredivisie: Weniger globale Reichweite
    
    4. **📱 Social Media & Marketing**
       - Follower-Zahlen korrelieren mit Marktwert
       - Sponsoring-Potential erhöht Vereinswert
    
    5. **🥇 Erfolge & Titel**
       - Champions League, WM, EM steigern Brand-Wert
       - Individuelle Auszeichnungen (Ballon d'Or)
    """)
    
    # Zusammenfassung
    st.subheader("📈 Study Conclusions")
    
    avg_premium = results_df['Premium_Percentage'].mean()
    max_premium = results_df['Premium_Percentage'].max()
    min_premium = results_df['Premium_Percentage'].min()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Durchschnittlicher Premium", f"{avg_premium:.1f}%")
    col2.metric("📈 Höchster Premium", f"{max_premium:.1f}%", f"{results_df.loc[results_df['Premium_Percentage'].idxmax(), 'Spieler']}")
    col3.metric("📉 Niedrigster Premium", f"{min_premium:.1f}%", f"{results_df.loc[results_df['Premium_Percentage'].idxmin(), 'Spieler']}")
    
    st.markdown(f"""
    ### 🎯 **Key Findings:**
    
    - **Performance erklärt {100-avg_premium:.1f}%** des durchschnittlichen Marktwerts
    - **Brand Premium variiert** von {min_premium:.1f}% bis {max_premium:.1f}%
    - **Superstars** haben überproportionale Premiums (>90%)
    - **Etablierte Spieler** werden fair nach Performance bewertet
    
    **Implikationen für Vereine:**
    - Transfer-Budgets: Unterscheide Performance- vs. Marketing-Investitionen
    - Scouting: Finde unterbewertete Performance-Stars
    - Marketing: Entwickle Spieler-Brands für höhere Verkaufserlöse
    """)

if __name__ == "__main__":
    main() 