import os
import streamlit as st
import generate as gen

st.set_page_config(
    page_title="TurkLM - Türkçe GPT Metin Üreteci",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Görsel 2 ile %100 Birebir Aynı Açık Tema (Light Mode) CSS Tasarımı
custom_css = """
<style>
    /* Dark mode override & zemin rengi */
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(248, 250, 252, 0.8) !important;
    }

    /* Üst Banner Card */
    .header-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);
    }
    
    .badge {
        display: inline-block;
        background-color: #fff1f2;
        color: #7c1034;
        border: 1px solid #fecdd3;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        margin-bottom: 1rem;
        letter-spacing: 0.06em;
    }
    
    .title-text {
        color: #252f9c;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }
    
    .subtitle-text {
        color: #475569;
        font-size: 1.02rem;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Beyaz Kart Konteynerleri */
    .card-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }

    /* Metin Kutuları (Text Area) */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-size: 0.98rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #252f9c !important;
        box-shadow: 0 0 0 3px rgba(37, 47, 156, 0.12) !important;
    }

    /* Örnek Butonları (Light Gray Pill) */
    .stButton>button {
        background-color: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #e2e8f0 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* Ana Çalıştır Butonu */
    .main-btn button {
        background: #252f9c !important;
        color: #ffffff !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 1.5rem !important;
        margin-top: 1rem !important;
        box-shadow: 0 4px 14px rgba(37, 47, 156, 0.3) !important;
    }
    
    .main-btn button:hover {
        background: #7c1034 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 18px rgba(124, 16, 52, 0.35) !important;
    }
    
    /* Etiketler ve Başlıklar */
    .card-label {
        color: #1e293b;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# Model yükleme işlemini önbelleğe alma
@st.cache_resource
def get_model():
    checkpoint_path = 'checkpoints/model.pt'
    if not os.path.exists(checkpoint_path):
        return None, None, None
    try:
        return gen.load_model(checkpoint_path)
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None, None


model, tokenizer, device = get_model()

# 1. Üst Banner Kartı (Görsel 2 ile Birebir)
st.markdown(
    """
    <div class="header-box">
        <span class="badge">KARAKTER BAZLI GPT v1.0</span>
        <div class="title-text">TurkLM Metin Üreteci</div>
        <div class="subtitle-text">Yapay zeka modelinize bir başlangıç cümlesi verin, mantıksal akışa uygun olarak hikayeyi ve metni devam ettirsin.</div>
    </div>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.warning("Model checkpoint dosyası (`checkpoints/model.pt`) bulunamadı veya yüklenemedi.")

# Prompt Yönetimi (Session State)
if "prompt_value" not in st.session_state:
    st.session_state.prompt_value = "Yapay zeka teknolojisi gelecekte..."

# 2. İki Kolonlu Kart Yapısı
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card-label">Başlangıç Metni (Prompt)</div>', unsafe_allow_html=True)
    prompt_input = st.text_area(
        label="Prompt",
        label_visibility="collapsed",
        value=st.session_state.prompt_value,
        height=140
    )
    
    st.markdown('<div class="card-label" style="margin-top: 1.2rem;">Örnek Başlangıç Cümleleri</div>', unsafe_allow_html=True)
    ex1, ex2 = st.columns(2)
    if ex1.button("Yapay zeka teknolojileri"):
        st.session_state.prompt_value = "Yapay zeka teknolojileri günümüzde"
        st.rerun()
    if ex2.button("Bir zamanlar uzak bir ülkede"):
        st.session_state.prompt_value = "Bir zamanlar uzak bir ülkede"
        st.rerun()
        
    ex3, ex4 = st.columns(2)
    if ex3.button("Bilim insanları yeni araştırmada"):
        st.session_state.prompt_value = "Bilim insanları yeni araştırmada"
        st.rerun()
    if ex4.button("Türkiye'nin tarihi güzellikleri"):
        st.session_state.prompt_value = "Türkiye'nin tarihi güzellikleri"
        st.rerun()

    with st.expander("Gelişmiş Parametreler", expanded=False):
        temperature = st.slider(
            "Temperature (Yaratıcılık Düzeyi)",
            min_value=0.1, max_value=1.5, value=0.8, step=0.05
        )
        max_tokens = st.slider(
            "Maksimum Karakter Sayısı",
            min_value=50, max_value=1000, value=300, step=50
        )

    st.markdown('<div class="main-btn">', unsafe_allow_html=True)
    run_btn = st.button("Metni Devam Ettir")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-label">Model Tarafından Üretilen Metin</div>', unsafe_allow_html=True)
    output_container = st.empty()
    output_container.text_area(
        label="Sonuc",
        label_visibility="collapsed",
        value="",
        height=380,
        disabled=True
    )

if run_btn:
    if not prompt_input or not prompt_input.strip():
        st.warning("Lütfen bir başlangıç metni girin.")
    elif model is None:
        st.error("Model yüklenmedi.")
    else:
        with st.spinner("TurkLM metni üretiyor..."):
            try:
                result = gen.generate(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=prompt_input,
                    max_tokens=int(max_tokens if 'max_tokens' in locals() else 300),
                    temperature=float(temperature if 'temperature' in locals() else 0.8)
                )
                output_container.text_area(
                    label="Sonuc",
                    label_visibility="collapsed",
                    value=result,
                    height=380
                )
            except Exception as e:
                st.error(f"Üretim sırasında hata oluştu: {e}")
