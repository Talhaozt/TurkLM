import os
import streamlit as st
import generate as gen

st.set_page_config(
    page_title="TurkLM - Türkçe GPT Metin Üreteci",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel Stil (HEX Renk Paleti: #cfe5ff, #7c1034, #252f9c)
custom_css = """
<style>
    .stApp {
        background-color: #f0f6ff;
    }
    .main-title {
        color: #252f9c;
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #7c1034;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #252f9c !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(37, 47, 156, 0.25) !important;
    }
    .stButton>button:hover {
        background-color: #7c1034 !important;
        box-shadow: 0 6px 16px rgba(124, 16, 52, 0.35) !important;
    }
    div[data-baseweb="select"] {
        border-color: #cfe5ff !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# Model yükleme işlemini önbelleğe (cache) alma
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

# Başlık Paneli
st.markdown('<div class="main-title">TurkLM Metin Üreteci</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Karakter Bazlı Türkçe GPT v1.0</div>', unsafe_allow_html=True)

if model is None:
    st.warning("Model checkpoint dosyası (`checkpoints/model.pt`) bulunamadı veya yüklenemedi.")

# Sol Yan Menü (Parametreler)
with st.sidebar:
    st.header("⚙️ Gelişmiş Ayarlar")
    temperature = st.slider(
        "Temperature (Yaratıcılık Düzeyi)",
        min_value=0.1,
        max_value=1.5,
        value=0.8,
        step=0.05,
        help="Düşük değerler mantıklı ve kararlı, yüksek değerler yaratıcı sonuçlar üretir."
    )
    max_tokens = st.slider(
        "Maksimum Karakter Sayısı",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
        help="Modelin üreteceği maksimum karakter uzunluğu."
    )
    
    st.markdown("---")
    st.markdown("**Model Mimarisi:**")
    st.markdown("- Embedding: 384\n- Heads: 6\n- Layers: 6\n- Block Size: 256")

# Ana İçerik Alanı
col1, col2 = st.columns(2)

with col1:
    st.subheader("Başlangıç Metni (Prompt)")
    prompt = st.text_area(
        label="Prompt",
        label_visibility="collapsed",
        placeholder="Örn: Yapay zeka teknolojisi gelecekte...",
        height=220
    )
    
    # Örnek Butonları
    st.markdown("**Hızlı Örnekler:**")
    ex_col1, ex_col2 = st.columns(2)
    if ex_col1.button("Yapay zeka teknolojileri"):
        prompt = "Yapay zeka teknolojileri günümüzde"
    if ex_col2.button("Bir zamanlar uzak bir ülkede"):
        prompt = "Bir zamanlar uzak bir ülkede"

with col2:
    st.subheader("Modelin Ürettiği Sonuç")
    output_container = st.empty()
    output_container.text_area(label="Sonuc", label_visibility="collapsed", value="", height=280, disabled=True)

# Üretim Butonu
if st.button("Metni Devam Ettir"):
    if not prompt or not prompt.strip():
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
                    prompt=prompt,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature)
                )
                output_container.text_area(label="Sonuc", label_visibility="collapsed", value=result, height=280)
            except Exception as e:
                st.error(f"Üretim sırasında hata oluştu: {e}")
