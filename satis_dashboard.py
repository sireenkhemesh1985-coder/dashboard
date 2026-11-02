import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# Sayfa Ayarları
st.set_page_config(page_title="Akıllı AI Satış Analisti", layout="wide")

st.sidebar.title("🤖 AI Analist Ayarları")
api_key = st.sidebar.text_input("Google Gemini API Key Girin:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    try:
        # En güncel sürüm çağrısı ile modeli tanımlıyoruz
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    except Exception:
        # Alternatif olarak kütüphane eski sürümse bu formatı denetiyoruz
        model = genai.GenerativeModel('models/gemini-1.5-flash')


st.title("📈 Gelişmiş AI Destekli Satış Dashboard'u")
st.write("Excel verileriniz üzerinden düşen ürünleri, bölgesel zayıflıkları ve gelecek ay tahminlerini yapay zekaya analiz ettirin.")

uploaded_file = st.file_uploader("Excel veya CSV Dosyanızı Seçin", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
        
    st.success("Dosya başarıyla yüklendi!")
    
    # Kolaylık olsun diye tarih sütununu datetime formatına çevirelim
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sütun Eşleştirme")
    tarih_col = st.sidebar.selectbox("Tarih Sütunu (Yıl-Ay-Gün):", df.columns)
    ciro_col = st.sidebar.selectbox("Ciro/Tutar Sütunu:", df.columns)
    urun_col = st.sidebar.selectbox("Ürün/Kategori Sütunu:", df.columns)
    bolge_col = st.sidebar.selectbox("Bölge/Şehir Sütunu (Varsa):", df.columns)

    # Tarih formatını güvenli hale getir
    df[tarih_col] = pd.to_datetime(df[tarih_col])
    df['YılAy'] = df[tarih_col].dt.to_period('M').astype(str)

    # Grafikler
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Aylık Toplam Ciro Trendi")
        aylik_df = df.groupby('YılAy')[ciro_col].sum().reset_index()
        fig_line = px.line(aylik_df, x='YılAy', y=ciro_col, markers=True, template="plotly_dark")
        st.plotly_chart(fig_line, use_container_width=True)
    with g2:
        st.subheader("Bölgelere Göre Satış Dağılımı")
        bolge_df = df.groupby(bolge_col)[ciro_col].sum().reset_index()
        fig_bar = px.bar(bolge_df, x=bolge_col, y=ciro_col, color=bolge_col, template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==============================================================================
    # 🧠 ÖZELLEŞTİRİLMİŞ YAPAY ZEKA SKILL ALANI
    # ==============================================================================
    st.markdown("---")
    st.subheader("🔍 Yapay Zeka Derin Analiz İstasyonu")
    
    if not api_key:
        st.info("💡 Yapay zekanın nokta atışı analiz yapabilmesi için lütfen sol panelden Gemini API anahtarınızı girin.")
    else:
        if st.button("🚀 Derin Analiz Raporu Üret (Tahmin, Risk ve Zayıflıklar)"):
            with st.spinner("Yapay zeka verilerinizdeki düşüş trendlerini inceliyor ve gelecek ayı tahmin ediyor..."):
                try:
                    # AI için gelişmiş veri hazırlığı
                    # 1. Son 2 ayın ürün bazlı satışları (Düşen ürünleri bulmak için)
                    aylar = sorted(df['YılAy'].unique())
                    son_iki_ay_veri = ""
                    if len(aylar) >= 2:
                        son_ay = aylar[-1]
                        onceki_ay = aylar[-2]
                        urun_karsilastirma = df[df['YılAy'].isin([onceki_ay, son_ay])].groupby(['YılAy', urun_col])[ciro_col].sum().unstack(level=0).fillna(0)
                        son_iki_ay_veri = urun_karsilastirma.to_string()

                    # 2. Bölgesel performans verisi
                    bolge_ozeti = df.groupby([bolge_col])[ciro_col].agg(['sum', 'count']).to_string()
                    
                    # 3. Genel ciro trendi
                    ciro_trendi = aylik_df.to_string()

                    # 🎯 YAPAY ZEKAYA GİDEN ÖZEL TALİMAT (PROMPT)
                    prompt = f"""
                    Sen kıdemli bir veri analitiği uzmanı ve satış danışmanısın. 
                    Sana bir şirketin ham satış verilerinden türetilmiş kritik tabloları veriyorum:
                    
                    [Zaman Serisi Ciro Trendi (Aylık)]:
                    {ciro_trendi}
                    
                    [Bölgesel Satış ve Sipariş Özeti]:
                    {bolge_ozeti}
                    
                    [Son Dönem Ürün Bazlı Ciro Değişimleri]:
                    {son_iki_ay_veri}
                    
                    Senden bu verileri titizlikle inceleyip şu 3 ana başlık altında rapor hazırlamanı istiyorum:
                    
                    1. 📉 DÜŞEN ÜRÜNLERİN TESPİTİ: Son dönemde satışı veya cirosu gözle görülür şekilde azalan, alarm veren ürünleri/kategorileri listele. Bu düşüşün olası nedenlerini yorumla.
                    2. 🗺️ BÖLGESEL ZAYIFLIKLARIN BULUNMASI: Hangi bölgeler/şehirler ciro veya sipariş adeti olarak beklentinin altında kalmış? Potansiyeli olmasına rağmen zayıf kalan yerleri ve buraları canlandırma formüllerini yaz.
                    3. 🔮 GELECEK AYIN SATIŞ TAHMİNİ VE STRATEJİSİ: Mevcut trend çizgisine bakarak önümüzdeki ayın cirosu sence ne olur? (Matematiksel ve mantıksal bir tahminde bulun). Bu tahmini tutturmak veya aşmak için atılması gereken en kritik 3 adımı belirt.
                    
                    Raporu tamamen Türkçe, profesyonel bir tonda, maddeler halinde ve anlaşılır başlıklarla yaz.
                    """
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown("### 📋 Yapay Zeka Stratejik Analiz Raporu")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Analiz üretilirken teknik bir hata oluştu: {e}")
