"""
Interface Streamlit - Agent Support IT
Style ChatGPT/Claude avec input en bas
"""

import streamlit as st
import requests
import re

def normalize_response_text(text: str) -> str:
    """Normalise le texte de réponse pour réduire l'espacement vertical excessif"""
    if not text:
        return text
    # Remplacer 3+ newlines par 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Convertir les simples newlines en <br>, doubles en paragraphes
    paragraphs = text.split('\n\n')
    html_parts = []
    for p in paragraphs:
        if p.strip():
            # Remplacer les newlines simples par <br> dans chaque paragraphe
            p_html = p.strip().replace('\n', '<br>')
            html_parts.append(f"<p>{p_html}</p>")
    return ''.join(html_parts)

# Configuration de la page
st.set_page_config(
    page_title="Agent Support IT",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS bleu foncé style chat
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Fond bleu foncé */
    .stApp {
        background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Header compact */
    .header {
        text-align: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.5rem;
    }

    .title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Réduire espacement Streamlit par défaut */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }

    .stMarkdown {
        margin-bottom: 0 !important;
    }

    /* Zone de chat */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 0.5rem;
    }

    /* Message utilisateur */
    .user-message {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%);
        border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        margin-left: 15%;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .user-label {
        font-size: 0.75rem;
        color: #64b5f6;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* Message assistant */
    .assistant-message {
        background: linear-gradient(135deg, #132238 0%, #1a2f4a 100%);
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        margin-right: 5%;
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .assistant-label {
        font-size: 0.75rem;
        color: #4fc3f7;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    /* Classification badges */
    .badges {
        display: flex;
        gap: 0.4rem;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
    }

    .badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-queue {
        background: rgba(100, 181, 246, 0.2);
        color: #64b5f6;
        border: 1px solid rgba(100, 181, 246, 0.3);
    }

    .badge-high {
        background: rgba(244, 67, 54, 0.2);
        color: #ef5350;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }

    .badge-medium {
        background: rgba(255, 167, 38, 0.2);
        color: #ffa726;
        border: 1px solid rgba(255, 167, 38, 0.3);
    }

    .badge-low {
        background: rgba(102, 187, 106, 0.2);
        color: #66bb6a;
        border: 1px solid rgba(102, 187, 106, 0.3);
    }

    .response-text {
        line-height: 1.4;
        font-size: 0.9rem;
    }

    .response-text p {
        margin-bottom: 0.3rem;
    }

    /* Feedback buttons */
    .feedback-section {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Input area - fixed at bottom */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: #0a1628;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stChatInput > div {
        max-width: 800px;
        margin: 0 auto;
    }

    [data-testid="stChatInput"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 24px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #e2e8f0 !important;
    }

    [data-testid="stChatInput"] button {
        background: #1e88e5 !important;
        border-radius: 50% !important;
    }

    /* Spacing for fixed input */
    .main-content {
        padding-bottom: 100px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Buttons in columns */
    .stButton > button {
        background: rgba(30, 136, 229, 0.2) !important;
        color: #64b5f6 !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 20px !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: rgba(30, 136, 229, 0.4) !important;
        border-color: #64b5f6 !important;
    }

    /* Success/info messages */
    .stSuccess, .stInfo {
        background: rgba(30, 136, 229, 0.1) !important;
        border: 1px solid rgba(100, 181, 246, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #1e88e5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Configuration API depuis variable d'environnement
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialiser l'historique de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'last_prediction_id' not in st.session_state:
    st.session_state.last_prediction_id = None

# Header
st.markdown("""
<div class="header">
    <div class="title">Agent Support IT</div>
</div>
""", unsafe_allow_html=True)

# Container principal avec padding pour l'input fixe
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Afficher l'historique des messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <div class="user-label">VOUS</div>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Message assistant avec badges
        urgency = msg.get("urgency", "").lower()
        urgency_class = f"badge-{urgency}" if urgency in ["high", "medium", "low"] else "badge-low"

        st.markdown(f"""
        <div class="assistant-message">
            <div class="assistant-label">SUPPORT IT</div>
            <div class="badges">
                <span class="badge badge-queue">{msg.get("queue", "")}</span>
                <span class="badge {urgency_class}">{msg.get("urgency", "").upper()}</span>
            </div>
            <div class="response-text">{normalize_response_text(msg["content"])}</div>
        </div>
        """, unsafe_allow_html=True)

        # Feedback buttons pour le dernier message
        if msg == st.session_state.messages[-1] and st.session_state.last_prediction_id:
            col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
            with col1:
                if st.button("👍", key=f"up_{st.session_state.last_prediction_id}"):
                    st.success("Merci !")
            with col2:
                if st.button("👎", key=f"down_{st.session_state.last_prediction_id}"):
                    st.session_state.show_feedback_form = True

            # Formulaire de feedback si l'utilisateur a cliqué 👎
            if st.session_state.get("show_feedback_form"):
                with st.expander("Corriger la classification", expanded=True):
                    correct_queue = st.selectbox(
                        "Queue correcte",
                        options=["", "network", "hardware", "software", "security", "printer", "other"],
                        key=f"queue_{st.session_state.last_prediction_id}"
                    )
                    correct_urgency = st.selectbox(
                        "Urgence correcte",
                        options=["", "low", "medium", "high", "critical"],
                        key=f"urgency_{st.session_state.last_prediction_id}"
                    )
                    if st.button("Envoyer le feedback", key=f"submit_{st.session_state.last_prediction_id}"):
                        if correct_queue or correct_urgency:
                            try:
                                requests.post(
                                    f"{API_URL}/feedback",
                                    json={
                                        "prediction_id": st.session_state.last_prediction_id,
                                        "correct_queue": correct_queue if correct_queue else None,
                                        "correct_urgency": correct_urgency if correct_urgency else None
                                    },
                                    timeout=5
                                )
                                st.success("Feedback enregistré, merci !")
                                st.session_state.show_feedback_form = False
                            except Exception as e:
                                st.error(f"Erreur: {e}")
                        else:
                            st.warning("Veuillez sélectionner au moins une correction")

        # Afficher les sources RAG pour TOUS les messages assistant
        rag_sources = msg.get("rag_sources", [])
        if rag_sources:
            with st.expander("📚 Sources RAG utilisées", expanded=False):
                for i, src in enumerate(rag_sources, 1):
                    # Tronquer si trop long
                    truncated = src[:500] + "..." if len(src) > 500 else src
                    st.markdown(f"**{i}.** {truncated}")

st.markdown('</div>', unsafe_allow_html=True)

# Message d'accueil si pas de messages
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #64b5f6;">
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">Bienvenue sur le support IT</p>
        <p style="color: #90a4ae; font-size: 0.9rem;">Décrivez votre problème technique ci-dessous</p>
    </div>
    """, unsafe_allow_html=True)

# Input chat en bas
user_input = st.chat_input("Décrivez votre problème technique...")

if user_input:
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Détecter les messages simples qui ne nécessitent pas l'API
    user_lower = user_input.lower().strip()

    # Mots-clés de remerciement/confirmation
    gratitude_keywords = ["merci", "thanks", "thank you", "thx"]
    confirmation_keywords = ["ok", "d'accord", "compris", "entendu", "noté"]
    positive_keywords = ["super", "cool", "parfait", "top", "nickel", "genial", "génial", "excellent", "impec"]
    success_keywords = ["ca marche", "ça marche", "c'est bon", "c bon", "ca fonctionne", "ça fonctionne", "resolu", "résolu"]

    simple_response = None
    if any(kw in user_lower for kw in gratitude_keywords):
        simple_response = "Je vous en prie ! N'hésitez pas si vous avez d'autres questions techniques."
    elif any(kw in user_lower for kw in success_keywords):
        simple_response = "Parfait, ravi d'avoir pu résoudre votre problème ! Bonne continuation."
    elif any(kw in user_lower for kw in positive_keywords):
        simple_response = "Content d'avoir pu vous aider ! À votre disposition pour toute autre question."
    elif any(kw in user_lower for kw in confirmation_keywords) and len(user_lower) < 20:
        simple_response = "Très bien ! Contactez-nous si le problème persiste."

    if simple_response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": simple_response,
            "queue": "Suivi",
            "urgency": "low"
        })
        st.rerun()

    # Construire l'historique de conversation pour l'API (sans le message actuel)
    conversation_history = []
    for msg in st.session_state.messages[:-1]:  # Exclure le message qu'on vient d'ajouter
        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Appeler l'API avec l'historique
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "user_query": user_input,
                    "conversation_history": conversation_history if conversation_history else None
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Ajouter la réponse assistant
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get('response', 'Pas de réponse disponible.'),
                    "queue": result.get('predicted_queue', ''),
                    "urgency": result.get('predicted_urgency', ''),
                    "rag_sources": result.get('rag_sources', [])
                })

                st.session_state.last_prediction_id = result.get('prediction_id')
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Erreur serveur ({response.status_code}). Veuillez réessayer.",
                    "queue": "Erreur",
                    "urgency": "high"
                })

        except requests.exceptions.Timeout:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Le serveur met trop de temps à répondre. Veuillez réessayer.",
                "queue": "Timeout",
                "urgency": "medium"
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Erreur de connexion : {str(e)}",
                "queue": "Erreur",
                "urgency": "high"
            })

    # Rerun pour afficher les nouveaux messages
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### Technologies")
    st.markdown("""
    - XGBoost (classification)
    - Sentence Transformers
    - PGVector (RAG)
    - Mistral API (LLM)
    """)

    st.markdown("---")
    st.markdown("### Pipeline MLOps")
    st.markdown("""
    - Airflow
    - MLflow
    - Evidently
    - K3s + GitHub Actions
    """)

    st.markdown("---")
    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.session_state.last_prediction_id = None
        st.rerun()

    st.markdown("---")
    st.markdown(f"[API Docs]({API_URL}/docs)")
