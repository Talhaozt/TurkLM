import os
import gradio as gr
from generate import load_model, generate

# Model ve tokenizer yükleme
CHECKPOINT_PATH = 'checkpoints/model.pt'

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Uyarı: '{CHECKPOINT_PATH}' bulunamadı.")

try:
    print("Model yükleniyor...")
    model, tokenizer, device = load_model(CHECKPOINT_PATH)
    print(f"Model başarıyla yüklendi! Cihaz: {device}")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    model, tokenizer, device = None, None, None


def predict(prompt: str, max_tokens: int, temperature: float):
    if not prompt or not prompt.strip():
        return "Lütfen geçerli bir başlangıç metni (prompt) girin."

    if model is None:
        return "Hata: Model henüz yüklenmedi veya checkpoint dosyası bulunamadı."

    try:
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        return generated
    except Exception as e:
        return f"Üretim sırasında bir hata oluştu: {str(e)}"


# GÜNCELLENMİŞ CSS: SLIDER SAYI KUTULARINI DÜZELTME VE İYİLEŞTİRME
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root, .dark, body {
    color-scheme: light !important;
    --body-background-fill: #f6f8fc !important;
    --block-background-fill: #ffffff !important;
    --panel-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #e2e8f0 !important;
    --border-color-accent: #252f9c !important;
    --body-text-color: #0f172a !important;

    /* Gradio default slider vars */
    --input-background-fill: #ffffff !important;
    --input-text-color: #0f172a !important;
}

body, .gradio-container {
    background-color: #f6f8fc !important;
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

footer { visibility: hidden !important; }

.icon-inline {
    display: inline-flex;
    align-items: center;
    vertical-align: middle;
    margin-right: 6px;
}

/* ==========================================================================
   MODERN HEADER DESIGN
   ========================================================================== */
.header-card {
    position: relative;
    background: linear-gradient(135deg, #ffffff 0%, #edf4ff 50%, #fcf0f5 100%) !important;
    border: 1px solid rgba(37, 47, 156, 0.15) !important;
    border-radius: 24px !important;
    padding: 3rem 2rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 20px 40px -15px rgba(37, 47, 156, 0.08), 0 0 0 1px rgba(255, 255, 255, 0.8) inset !important;
    overflow: hidden;
}

.header-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 200px;
    background: radial-gradient(circle, rgba(37, 47, 156, 0.12) 0%, rgba(255,255,255,0) 70%);
    pointer-events: none;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    color: #7c1034;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.45rem 1.1rem;
    border-radius: 9999px;
    margin-bottom: 1.25rem;
    letter-spacing: 0.06em;
    border: 1px solid rgba(124, 16, 52, 0.2);
    box-shadow: 0 2px 8px rgba(124, 16, 52, 0.08);
}

.header-badge-dot {
    width: 7px;
    height: 7px;
    background-color: #7c1034;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px #7c1034;
}

.header-title {
    background: linear-gradient(135deg, #252f9c 0%, #7c1034 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.75rem;
    font-weight: 800;
    margin-bottom: 0.85rem;
    letter-spacing: -0.035em;
    line-height: 1.15;
}

.header-subtitle {
    color: #475569;
    font-size: 1.05rem;
    font-weight: 500;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ==========================================================================
   İÇERİK VE FORM ELEMANLARI
   ========================================================================== */
.custom-card {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.03) !important;
}

.block, .form, fieldset, .panel {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

textarea, input[type="text"] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    background-color: #ffffff !important;
    border-color: #252f9c !important;
    box-shadow: 0 0 0 4px rgba(37, 47, 156, 0.1) !important;
}

.block label span, [data-testid="block-label"] {
    background-color: transparent !important;
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
}

/* ==========================================================================
   YENİ: GÖRSELDEKİ SLIDER SAYI KUTULARINI DÜZELTEN CSS
   ========================================================================== */
input[type="number"][data-testid="slider-input"] {
    background-color: #ffffff !important; /* Koyu arka planı beyaz yapar */
    color: #1e293b !important; /* Sayı rengini koyu lacivert yapar */
    border: 1.5px solid #e2e8f0 !important; /* Yumuşak kenarlık ekler */
    border-radius: 8px !important; /* Hafif yuvarlatır */
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 6px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

/* Sayı kutusuna tıklandığında (focus) */
input[type="number"][data-testid="slider-input"]:focus {
    border-color: #252f9c !important;
    box-shadow: 0 0 0 3px rgba(37, 47, 156, 0.1) !important;
}

/* Slider çizgisinin rengini ayarlar */
.gr-slider div[role="slider"] {
    background-color: #252f9c !important;
}

/* Slider reset butonunu hizalar */
.gr-slider > div > button {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
    padding: 6px !important;
    transition: all 0.15s ease !important;
}

.gr-slider > div > button:hover {
    background-color: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
}
/* ========================================================================== */


.accordion {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    margin-top: 1rem !important;
}

/* Ana Buton */
.generate-btn {
    background: linear-gradient(135deg, #252f9c 0%, #1d247d 100%) !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.5rem !important;
    box-shadow: 0 8px 20px -4px rgba(37, 47, 156, 0.4) !important;
    transition: all 0.25 ease !important;
    margin-top: 1.25rem !important;
}

.generate-btn:hover {
    background: linear-gradient(135deg, #1d247d 0%, #7c1034 100%) !important;
    box-shadow: 0 12px 24px -4px rgba(124, 16, 52, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Örnek Çip Butonları */
.example-chip {
    background-color: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}

.example-chip:hover {
    background-color: #ffffff !important;
    border-color: #252f9c !important;
    color: #252f9c !important;
    box-shadow: 0 4px 12px rgba(37, 47, 156, 0.1) !important;
    transform: translateY(-1px);
}
"""

light_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "sans-serif"]
)

# SVG İkonlar
ICON_CPU = '<svg class="icon-inline" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="15" x2="23" y2="15"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="15" x2="4" y2="15"></line></svg>'
ICON_BOOK = '<svg class="icon-inline" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#252f9c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>'

with gr.Blocks(title="TurkLM - Türkçe GPT Metin Üreteci") as demo:
    # Header
    with gr.Row(elem_classes=["header-card"]):
        with gr.Column():
            gr.HTML(
                f"""
                <div class="header-badge">
                    <span class="header-badge-dot"></span>
                    {ICON_CPU} KARAKTER BAZLI GPT v1.0
                </div>
                <h1 class="header-title">TurkLM Metin Üreteci</h1>
                <p class="header-subtitle">Yapay zeka modelinize bir başlangıç cümlesi verin, mantıksal akışa uygun olarak hikayeyi ve metni devam ettirsin.</p>
                """
            )

    # Main Content
    with gr.Row(equal_height=True):
        # Sol Panel
        with gr.Column(scale=1, elem_classes=["custom-card"]):
            prompt_input = gr.Textbox(
                label="Başlangıç Metni (Prompt)",
                placeholder="Örn: Yapay zeka teknolojisi gelecekte...",
                lines=4
            )

            gr.HTML(
                f'<div style="font-weight: 700; font-size: 0.88rem; color: #1e293b; margin-top: 12px; margin-bottom: 8px;">{ICON_BOOK} Örnek Başlangıç Cümleleri</div>')

            with gr.Row():
                ex1 = gr.Button("Yapay zeka teknolojileri günümüzde", elem_classes=["example-chip"])
                ex2 = gr.Button("Bir zamanlar uzak bir ülkede", elem_classes=["example-chip"])
            with gr.Row():
                ex3 = gr.Button("Bilim insanları yeni araştırmada", elem_classes=["example-chip"])
                ex4 = gr.Button("Türkiye'nin tarihi güzellikleri", elem_classes=["example-chip"])

            # BURADA HATA OLAN SLIDERLAR YER ALIYOR
            with gr.Accordion("Gelişmiş Parametreler", open=False):
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.05,
                    label="Temperature (Yaratıcılık Düzeyi)"
                )
                max_tokens_slider = gr.Slider(
                    minimum=50, maximum=1000, value=300, step=50,
                    label="Maksimum Karakter Sayısı"
                )

            generate_btn = gr.Button(
                value="Metni Devam Ettir",
                variant="primary",
                size="lg",
                elem_classes=["generate-btn"]
            )

        # Sağ Panel
        with gr.Column(scale=1, elem_classes=["custom-card"]):
            output_text = gr.Textbox(
                label="Model Tarafından Üretilen Metin",
                lines=14,
                interactive=False
            )

    # Tıklama İşlevleri
    ex1.click(lambda: ("Yapay zeka teknolojileri günümüzde", 300, 0.8),
              outputs=[prompt_input, max_tokens_slider, temperature_slider])
    ex2.click(lambda: ("Bir zamanlar uzak bir ülkede", 250, 0.7),
              outputs=[prompt_input, max_tokens_slider, temperature_slider])
    ex3.click(lambda: ("Bilim insanları yeni yaptıkları araştırmada", 400, 0.8),
              outputs=[prompt_input, max_tokens_slider, temperature_slider])
    ex4.click(lambda: ("Türkiye'nin tarihi ve doğal güzellikleri", 300, 0.75),
              outputs=[prompt_input, max_tokens_slider, temperature_slider])

    generate_btn.click(
        fn=predict,
        inputs=[prompt_input, max_tokens_slider, temperature_slider],
        outputs=output_text
    )

if __name__ == '__main__':
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=light_theme,
        css=custom_css
    )