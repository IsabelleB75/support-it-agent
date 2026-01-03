"""
Interface Streamlit - Agent Support IT
Design moderne et simple
"""

import streamlit as st
import requests

# Configuration de la page
st.set_page_config(
    page_title="Agent Support IT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS moderne
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Fond avec gradient animé */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Header */
    .header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
    }

    .logo {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    .title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
    }

    /* Card principale */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Input styling */
    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #fff !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
    }

    /* Bouton principal */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.8rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }

    /* Résultats */
    .result-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    /* Classification cards */
    .class-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }

    .class-label {
        font-size: 0.8rem;
        color: #a0aec0;
        margin-bottom: 0.5rem;
    }

    .class-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.5rem;
    }

    .confidence {
        font-size: 0.75rem;
        color: #667eea;
    }

    /* Badges urgence */
    .urgency-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
    }

    .urgency-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #1a1a2e;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(247, 151, 30, 0.4);
    }

    .urgency-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }

    /* Réponse IA */
    .response-box {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #38ef7d;
        color: #e2e8f0;
        line-height: 1.7;
        font-size: 1rem;
    }

    /* Exemples */
    .example-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        color: #a0aec0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }

    .example-btn:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: #667eea;
        color: #fff;
    }

    /* Progress bar custom */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px;
    }

    /* Feedback buttons */
    .feedback-btn {
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.2s;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .footer a {
        color: #667eea;
        text-decoration: none;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Configuration API
API_URL = "http://78.47.129.250:30080"

# Header
st.markdown("""
<div class="header">
    <div class="logo">🤖</div>
    <div class="title">Agent Support IT</div>
    <div class="subtitle">Classification intelligente & réponse automatique</div>
</div>
""", unsafe_allow_html=True)

# Card principale
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# Zone de saisie
st.markdown('<p class="section-title">💬 Décrivez votre problème</p>', unsafe_allow_html=True)
user_query = st.text_area(
    label="Question",
    placeholder="Ex: Mon VPN ne fonctionne pas, je n'arrive pas à me connecter au réseau...",
    height=100,
    label_visibility="collapsed"
)

# Exemples rapides
st.markdown('<p style="color: #718096; font-size: 0.85rem; margin: 1rem 0 0.5rem 0;">Exemples rapides :</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌐 VPN", use_container_width=True):
        st.session_state.query = "Mon VPN ne fonctionne pas, je n'arrive pas à me connecter"
        st.rerun()
    if st.button("🖨️ Imprimante", use_container_width=True):
        st.session_state.query = "Mon imprimante n'imprime plus rien"
        st.rerun()

with col2:
    if st.button("💻 PC lent", use_container_width=True):
        st.session_state.query = "Mon ordinateur est très lent depuis ce matin"
        st.rerun()
    if st.button("🔐 Mot de passe", use_container_width=True):
        st.session_state.query = "J'ai oublié mon mot de passe"
        st.rerun()

with col3:
    if st.button("📧 Email", use_container_width=True):
        st.session_state.query = "Je ne reçois plus mes emails depuis hier"
        st.rerun()
    if st.button("🔧 Excel", use_container_width=True):
        st.session_state.query = "Excel plante à chaque ouverture de fichier"
        st.rerun()

# Utiliser la query de session si disponible
if 'query' in st.session_state and st.session_state.query:
    user_query = st.session_state.query
    st.session_state.query = None

st.markdown("<br>", unsafe_allow_html=True)

# Bouton analyser
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🚀 Analyser ma demande", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Traitement
if analyze_btn and user_query:

    with st.spinner(""):
        # Loader custom
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem; animation: float 1s ease-in-out infinite;">🔍</div>
            <p style="color: #a0aec0; margin-top: 1rem;">Analyse en cours...</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"user_query": user_query},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Section Résultats
                st.markdown("---")
                st.markdown('<p class="section-title">📊 Résultats de l\'analyse</p>', unsafe_allow_html=True)

                # Classification et Urgence
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="class-card">', unsafe_allow_html=True)
                    st.markdown('<p class="class-label">📁 CATÉGORIE</p>', unsafe_allow_html=True)
                    queue = result.get('predicted_queue', 'N/A')
                    st.markdown(f'<p class="class-value">{queue}</p>', unsafe_allow_html=True)

                    conf_q = result.get('confidence_queue', 0)
                    if conf_q:
                        st.progress(conf_q)
                        st.markdown(f'<p class="confidence">Confiance : {conf_q*100:.0f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="class-card">', unsafe_allow_html=True)
                    st.markdown('<p class="class-label">⚡ URGENCE</p>', unsafe_allow_html=True)

                    urgency = result.get('predicted_urgency', 'N/A').lower()
                    if urgency == 'high':
                        st.markdown('<span class="urgency-high">🔴 HIGH</span>', unsafe_allow_html=True)
                    elif urgency == 'medium':
                        st.markdown('<span class="urgency-medium">🟠 MEDIUM</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="urgency-low">🟢 LOW</span>', unsafe_allow_html=True)

                    conf_u = result.get('confidence_urgency', 0)
                    if conf_u:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.progress(conf_u)
                        st.markdown(f'<p class="confidence">Confiance : {conf_u*100:.0f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Réponse
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="section-title">💡 Réponse de l\'assistant</p>', unsafe_allow_html=True)

                ai_response = result.get('response', 'Pas de réponse disponible.')
                st.markdown(f'<div class="response-box">{ai_response}</div>', unsafe_allow_html=True)

                # Sources RAG
                sources = result.get('rag_sources', [])
                if sources:
                    with st.expander("📚 Sources utilisées"):
                        for src in sources:
                            st.markdown(f"• {src}")

                # Feedback
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p style="color: #718096; font-size: 0.9rem;">Cette réponse vous a-t-elle aidé ?</p>', unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 1, 2])
                pred_id = result.get('prediction_id')

                with col1:
                    if st.button("👍 Oui", use_container_width=True):
                        try:
                            requests.post(f"{API_URL}/feedback", json={"prediction_id": pred_id, "helpful": True}, timeout=5)
                        except:
                            pass
                        st.success("✨ Merci pour votre retour !")

                with col2:
                    if st.button("👎 Non", use_container_width=True):
                        try:
                            requests.post(f"{API_URL}/feedback", json={"prediction_id": pred_id, "helpful": False}, timeout=5)
                        except:
                            pass
                        st.info("📝 Merci, nous allons nous améliorer !")

            else:
                st.error(f"❌ Erreur serveur : {response.status_code}")

        except requests.exceptions.Timeout:
            st.error("⏱️ Le serveur met trop de temps à répondre.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Impossible de se connecter au serveur.")
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

elif analyze_btn and not user_query:
    st.warning("⚠️ Veuillez décrire votre problème.")

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Technologies")
    st.markdown("""
    - **Classification** : XGBoost
    - **NLP** : Sentence Transformers
    - **RAG** : PGVector
    - **LLM** : Mistral API
    """)

    st.markdown("---")
    st.markdown("## 📊 Pipeline MLOps")
    st.markdown("""
    - Airflow (orchestration)
    - MLflow (tracking)
    - Evidently (monitoring)
    - K3s (déploiement)
    - GitHub Actions (CI/CD)
    """)

    st.markdown("---")
    st.markdown("## 🔗 Liens")
    st.markdown(f"[📖 API Swagger]({API_URL}/docs)")
    st.markdown("[💻 GitHub](https://github.com/IsabelleB75/support-it-agent)")

# Footer
st.markdown("""
<div class="footer">
    <p>🎓 Projet MLOps - Bootcamp Jedha</p>
    <p>Isabelle B. | 2024</p>
</div>
""", unsafe_allow_html=True)
