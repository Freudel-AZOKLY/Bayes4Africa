import streamlit as st
import pandas as pd
import numpy as np
import requests
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import plotly.express as px
import logging
import smtplib
from email.mime.text import MIMEText

# --- Logging basique pour debug et monitoring ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# --- CSS personnalisé + animation + thème sombre optionnel ---
st.markdown("""
<style>
    .main > div { padding: 1rem 2rem; }
    h1, h2, h3 { color: #006400; }
    .css-1aumxhk {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    .css-1aumxhk:hover {
        background-color: #388E3C !important;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .fade-in {
        animation: fadeIn 1s ease-in;
    }
</style>
""", unsafe_allow_html=True)

# --- Fonctions utilitaires ---

@st.cache_data(show_spinner=False)
def fetch_worldbank_data(country_codes, indicator, start_year=2000, end_year=2024):
    records = []
    for cc in country_codes:
        try:
            url = f"https://api.worldbank.org/v2/country/{cc}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=100"
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                st.warning(f"Erreur chargement API pour {cc} ({r.status_code})")
                continue
            data = r.json()
            if data and len(data) > 1:
                for entry in data[1]:
                    if entry["value"] is not None:
                        records.append({
                            "country": cc,
                            "date": int(entry["date"]),
                            "value": float(entry["value"])
                        })
        except requests.exceptions.RequestException as e:
            st.warning(f"Erreur réseau pour {cc}: {e}")
            logging.error(f"Erreur API {cc}: {e}")
    df = pd.DataFrame(records)
    return df

def bayesian_multivariate_regression_with_forecast(df1, df2, forecast_years=3):
    x = df1["date"].values
    x_centered = x - x.mean()
    y1 = df1["value"].values
    y2 = df2["value"].values

    x_future = np.arange(x[-1] + 1, x[-1] + 1 + forecast_years)
    x_all = np.concatenate([x, x_future])
    x_all_centered = x_all - x.mean()

    with pm.Model() as model:
        x_shared = pm.Data("x_shared", x_centered)

        alpha1 = pm.Normal("alpha1", mu=0, sigma=10)
        beta1 = pm.Normal("beta1", mu=0, sigma=1)
        sigma1 = pm.Exponential("sigma1", 1)

        alpha2 = pm.Normal("alpha2", mu=0, sigma=10)
        beta2 = pm.Normal("beta2", mu=0, sigma=1)
        sigma2 = pm.Exponential("sigma2", 1)

        mu1 = alpha1 + beta1 * x_shared
        mu2 = alpha2 + beta2 * x_shared

        pm.Normal("y1_obs", mu=mu1, sigma=sigma1, observed=y1)
        pm.Normal("y2_obs", mu=mu2, sigma=sigma2, observed=y2)

        trace = pm.sample(1500, tune=1500, cores=1, progressbar=False, random_seed=42)

        pm.set_data({"x_shared": x_all_centered})

        pred1 = np.expand_dims(trace.posterior["alpha1"].values, -1) + \
                np.expand_dims(trace.posterior["beta1"].values, -1) * x_all_centered[None, None, :]

        pred2 = np.expand_dims(trace.posterior["alpha2"].values, -1) + \
                np.expand_dims(trace.posterior["beta2"].values, -1) * x_all_centered[None, None, :]

    return trace, x, x_all, pred1, pred2

def send_email(subject, message, to_email):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "ton.email@gmail.com"  # Modifier ici
    password = "ton_mdp_app"             # Modifier ici

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Erreur envoi email: {e}")
        logging.error(f"Erreur envoi email: {e}")
        return False

# --- Interface Streamlit ---

st.set_page_config(page_title="Veille Énergétique AFR", layout="wide", initial_sidebar_state="expanded")

st.title("Dashboard Veille Économique - Énergies Renouvelables Afrique Francophone")

# --- Données pays + indicateurs ---
pays_dispo = {
    "Bénin": "BJ", "Burkina Faso": "BF", "Burundi": "BI", "Cameroun": "CM",
    "République centrafricaine": "CF", "Comores": "KM", "Congo (Brazzaville)": "CG",
    "Congo (Kinshasa)": "CD", "Côte d'Ivoire": "CI", "Djibouti": "DJ", "Gabon": "GA",
    "Guinée": "GN", "Guinée-Bissau": "GW", "Guinée équatoriale": "GQ", "Madagascar": "MG",
    "Mali": "ML", "Maroc": "MA", "Mauritanie": "MR", "Niger": "NE", "Rwanda": "RW",
    "Sénégal": "SN", "Seychelles": "SC", "Tchad": "TD", "Togo": "TG", "Tunisie": "TN"
}

indicators = {
    "Part énergies renouvelables (%)": "EG.ELC.RNEW.ZS",
    "Production électricité renouvelable (kWh)": "EG.ELC.RNWX.KH"
}

# --- Sidebar ---

st.sidebar.header("Filtres")

selected_countries = st.sidebar.multiselect("Choisir pays", list(pays_dispo.keys()), default=["Sénégal", "Maroc"])

selected_indicators = st.sidebar.multiselect("Choisir indicateurs", list(indicators.keys()), default=list(indicators.keys()))

years = st.sidebar.slider("Années", 2000, 2024, (2018, 2024))

forecast_years = st.sidebar.slider("Années de prévision", 1, 10, 3)

export_requested = st.sidebar.button("Exporter données affichées (CSV)")

# --- Alertes & Notifications ---
st.sidebar.header("Alertes & Notifications")

enable_alerts = st.sidebar.checkbox("Activer alertes par email")

if enable_alerts:
    alert_indicator = st.sidebar.selectbox("Indicateur pour alerte", selected_indicators)
    alert_country = st.sidebar.selectbox("Pays pour alerte", selected_countries)
    alert_threshold = st.sidebar.number_input("Seuil d'alerte", min_value=0.0, step=0.1, value=50.0)
    email_to_alert = st.sidebar.text_input("Email de notification")
else:
    # Valeurs par défaut si alertes désactivées pour éviter erreurs
    alert_indicator = None
    alert_country = None
    alert_threshold = None
    email_to_alert = None

# --- Validation filtres ---
if not selected_countries:
    st.warning("Sélectionnez au moins un pays.")
    st.stop()

if not selected_indicators:
    st.warning("Sélectionnez au moins un indicateur.")
    st.stop()

# --- Récupération des codes pays ---
country_codes = [pays_dispo[p] for p in selected_countries]

# --- Chargement données ---
dfs = {}
with st.spinner("Chargement des données en cours..."):
    for ind in selected_indicators:
        df = fetch_worldbank_data(country_codes, indicators[ind], years[0], years[1])
        if df.empty:
            st.warning(f"Aucune donnée récupérée pour l'indicateur {ind} sur la période sélectionnée.")
        dfs[ind] = df

# --- Affichage tableaux et graphiques ---
for ind, df in dfs.items():
    st.subheader(f"Données World Bank : {ind}")
    if df.empty:
        st.write(f"Pas de données à afficher pour {ind}.")
        continue

    # Export CSV des données
    if export_requested:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"Télécharger CSV {ind}", data=csv_data, file_name=f"data_{ind}.csv", mime="text/csv")

    # Table interactive
    st.dataframe(df)

    # Graphique interactif
    fig = px.line(df, x="date", y="value", color="country", markers=True,
                  labels={"date": "Année", "value": ind, "country": "Pays"},
                  title=f"{ind} par pays de {years[0]} à {years[1]}")
    fig.update_layout(legend_title_text='Pays')
    st.plotly_chart(fig, use_container_width=True)

# --- Message d'information sur l'analyse bayésienne ---
if len(selected_countries) > 1:
    st.info(
        f"⚠️ L'analyse bayésienne multivariée avec prédiction future sera réalisée uniquement sur le premier pays sélectionné : {selected_countries[0]}."
    )

# --- Analyse bayésienne multivariée avec prédiction future ---
if len(selected_indicators) >= 2 and len(selected_countries) >= 1:
    ind1, ind2 = selected_indicators[:2]
    code = pays_dispo[selected_countries[0]]

    df1 = dfs[ind1][dfs[ind1]["country"] == code].sort_values("date") if "country" in dfs[ind1] else pd.DataFrame()
    df2 = dfs[ind2][dfs[ind2]["country"] == code].sort_values("date") if "country" in dfs[ind2] else pd.DataFrame()

    st.subheader(f"Analyse bayésienne multivariée avec prédiction future ({ind1} & {ind2}) sur {selected_countries[0]}")

    if df1.empty or df2.empty:
        st.warning("Pas assez de données pour l'analyse bayésienne.")
    else:
        with st.spinner("Exécution de l'analyse bayésienne..."):
            trace, x_obs, x_all, pred1_samples, pred2_samples = bayesian_multivariate_regression_with_forecast(df1, df2, forecast_years=forecast_years)

        # Calcul moyennes et HDI (intervalle crédibilité)
        pred1_mean = pred1_samples.mean(axis=(0, 1))
        pred1_hpd = az.hdi(pred1_samples.reshape(-1, pred1_samples.shape[-1]), hdi_prob=0.95)

        pred2_mean = pred2_samples.mean(axis=(0, 1))
        pred2_hpd = az.hdi(pred2_samples.reshape(-1, pred2_samples.shape[-1]), hdi_prob=0.95)

        # Graphique matplotlib amélioré
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(x_all, pred1_mean, label=f"Prédiction {ind1}", color="green")
        ax.fill_between(x_all, pred1_hpd[:, 0], pred1_hpd[:, 1], color="green", alpha=0.3)

        ax.plot(x_all, pred2_mean, label=f"Prédiction {ind2}", color="blue")
        ax.fill_between(x_all, pred2_hpd[:, 0], pred2_hpd[:, 1], color="blue", alpha=0.3)

        ax.scatter(x_obs, df1["value"], color="darkgreen", label=f"Observé {ind1}", zorder=5)
        ax.scatter(x_obs, df2["value"], color="darkblue", label=f"Observé {ind2}", zorder=5)

        ax.set_xlabel("Année")
        ax.set_ylabel("Valeur")
        ax.legend()
        ax.set_title("Analyse bayésienne multivariée avec prédiction future")
        ax.grid(True)
        st.pyplot(fig)

        # --- Tableau résumé ---
        last_year = max(df1["date"].max(), df2["date"].max())
        last_val1 = df1[df1["date"] == last_year]["value"].values[0] if last_year in df1["date"].values else np.nan
        last_val2 = df2[df2["date"] == last_year]["value"].values[0] if last_year in df2["date"].values else np.nan

        forecast_year = x_all[-1]
        pred1_forecast = pred1_samples.mean(axis=(0, 1))[-1]
        pred2_forecast = pred2_samples.mean(axis=(0, 1))[-1]

        summary_df = pd.DataFrame({
            "Élément": [
                "Pays analysé",
                "Indicateurs analysés",
                "Période analysée",
                "Dernière année observée",
                f"Dernière valeur observée ({ind1})",
                f"Dernière valeur observée ({ind2})",
                f"Prévision moyenne {forecast_year} ({ind1})",
                f"Prévision moyenne {forecast_year} ({ind2})",
                "Seuil alerte activé",
                "Seuil d'alerte",
                "Valeur actuelle dépasse seuil"
            ],
            "Valeur": [
                selected_countries[0],
                f"{ind1} et {ind2}",
                f"{years[0]} - {years[1]}",
                last_year,
                f"{last_val1:.2f}",
                f"{last_val2:.2f}",
                f"{pred1_forecast:.2f}",
                f"{pred2_forecast:.2f}",
                "Oui" if enable_alerts else "Non",
                alert_threshold if enable_alerts else "N/A",
                (
                    "Oui" if enable_alerts and not df1.empty and last_val1 > alert_threshold
                    else "Non"
                )
            ]
        })

        st.subheader("Résumé des informations clés")
        st.table(summary_df)

        # --- Diagnostics MCMC ---
        st.subheader("Diagnostics MCMC")
        st.write("Visualisation des chaînes de Markov (trace plots) pour les paramètres clés :")
        az_trace_figs = az.plot_trace(trace, var_names=["alpha1", "beta1", "alpha2", "beta2"], compact=True)

        if hasattr(az_trace_figs, "figure"):
            st.pyplot(az_trace_figs)
        else:
            for ax in np.array(az_trace_figs).flatten():
                st.pyplot(ax.figure)

        st.write("Valeurs R-hat (convergence) :")
        rhat = az.rhat(trace, var_names=["alpha1", "beta1", "alpha2", "beta2"])

        rhat_dict = {var: float(rhat[var].values) for var in rhat.data_vars}
        st.dataframe(pd.DataFrame.from_dict(rhat_dict, orient='index', columns=['R-hat']))

        st.write("Autocorrélations :")
        autocorr_figs = az.plot_autocorr(trace, var_names=["alpha1", "beta1", "alpha2", "beta2"])

        if hasattr(autocorr_figs, "figure"):
            st.pyplot(autocorr_figs)
        else:
            for ax in np.array(autocorr_figs).flatten():
                st.pyplot(ax.figure)

# --- Alertes email simples ---

if enable_alerts:
    if st.sidebar.button("Tester l'alerte email"):
        df_alert = dfs[alert_indicator]
        df_alert = df_alert[(df_alert["country"] == pays_dispo[alert_country]) & (df_alert["date"] == df_alert["date"].max())]
        if not df_alert.empty and df_alert["value"].values[0] > alert_threshold:
            success = send_email(
                subject=f"Alerte {alert_indicator} pour {alert_country}",
                message=f"Attention, la valeur de {alert_indicator} pour {alert_country} est de {df_alert['value'].values[0]}, dépassant le seuil {alert_threshold}.",
                to_email=email_to_alert
            )
            if success:
                st.sidebar.success("Email envoyé avec succès !")
        else:
            st.sidebar.info("Valeur actuelle en-dessous du seuil, pas d'alerte envoyée.")

# --- Fin ---
