import os
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

# Türkçe dilinin stop words'lerini indirin
import nltk
nltk.download('stopwords')

# Ses dosyalarının bulunduğu klasör yolu
klasor_yolu = "/home/firengiz/İndirilenler/audio/Audio_Eng"

# Dil sınıflandırma modelini yükleme
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Türkçe stop words'leri al
turkce_stopwords = set(stopwords.words('turkish'))

# Klasördeki tüm dosyaları al
dosya_listesi = os.listdir(klasor_yolu)

# Ses dosyalarını sırasıyla aç, transkript oluştur ve gerçek dil tahmini yap
for mp4_dosyasi in dosya_listesi:
    if mp4_dosyasi.endswith(".mp4"):
        dosya_yolu = os.path.join(klasor_yolu, mp4_dosyasi)

        # Ses dosyasını yükleme
        ses = AudioSegment.from_file(dosya_yolu, format="mp4")

        # Ses dosyasını WAV formatına dönüştürme
        wav_dosya_yolu = dosya_yolu.replace(".mp4", ".wav")
        ses.export(wav_dosya_yolu, format="wav")

        # Ses dosyasını Google Speech Recognition API kullanarak transkript oluşturma
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_dosya_yolu) as source:
            try:
                audio_data = recognizer.record(source)
                transkript = recognizer.recognize_google(audio_data, language="tr")
            except sr.UnknownValueError:
                transkript = ""  # Tanınamayan değer hatası durumunda transkripti boş bir dize olarak ayarla

        # Ses dosyasının transkripti ile gerçek dil tahmini yapma
        kelime_listesi = wordpunct_tokenize(transkript)
        turkce_kelime_sayisi = sum(1 for kelime in kelime_listesi if kelime.isalpha() and kelime.isascii() and kelime.lower() in turkce_stopwords)

        # Ses dosyasının transkripti ile dil tahmini yapma
        inputs = tokenizer(transkript, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()

        # Gerçek dil tahminini belirleme
        real_language = "Turkish" if "tr" in mp4_dosyasi else "English"

        # Ses dosyasını oynatma
        play(ses)

        language_labels = ["English", "Turkish"]
        # Türkçe veya İngilizce dil tahminini ve transkripti yazdırma
        if turkce_kelime_sayisi > 0:
            predicted_language = "Turkish"
        else:
            predicted_language = "English"

        print(f"{mp4_dosyasi} dosyasındaki gerçek dil: {real_language}")
        print(f"{mp4_dosyasi} dosyasındaki dil tahmini: {predicted_language}")
        print(f"Transkript: {transkript}")

        # Kullanıcıdan bir tuşa basılmasını bekleyin
        input(f"{mp4_dosyasi} dosyasını dinledikten sonra devam etmek için bir tuşa basın...")

print("İşlem tamamlandı.")




#############################################
# Ses'e Ses eklemek


# import os
# from pydub import AudioSegment
# from pydub.playback import play
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import speech_recognition as sr
# from nltk import wordpunct_tokenize
# from nltk.corpus import stopwords

# # Türkçe dilinin stop words'lerini indirin
# import nltk
# nltk.download('stopwords')

# # Ses dosyalarının bulunduğu klasör yolu
# klasor_yolu = "/home/firengiz/İndirilenler/audio/Audio_Eng"

# # Dil sınıflandırma modelini yükleme
# model_name = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Türkçe stop words'leri al
# turkce_stopwords = set(stopwords.words('turkish'))

# # Önceden kaydedilmiş "Merhaba" ses kaydını yükleme
# merhaba_ses = AudioSegment.from_file("/home/firengiz/İndirilenler/audio/Audio_Eng/merhaba.mp4", format="mp4")

# # Klasördeki tüm dosyaları al
# dosya_listesi = os.listdir(klasor_yolu)

# # Her ses dosyasının başına "Merhaba" kelimesini ekle
# for mp4_dosyasi in dosya_listesi:
#     if mp4_dosyasi.endswith(".mp4"):
#         dosya_yolu = os.path.join(klasor_yolu, mp4_dosyasi)

#         # Ses dosyasını yükleme
#         ses = AudioSegment.from_file(dosya_yolu, format="mp4")

#         # Ses dosyasının başına "Merhaba" kelimesini ekleme
#         yeni_ses = merhaba_ses + ses

#         # Yeni sesi oynat
#         play(yeni_ses)

#         # Ses dosyasını WAV formatına dönüştürme
#         wav_dosya_yolu = dosya_yolu.replace(".mp4", "_merhaba_eklenmis.wav")
#         yeni_ses.export(wav_dosya_yolu, format="wav")

#         # Ses dosyasının transkripti ile gerçek dil tahmini yapma
#         recognizer = sr.Recognizer()
#         with sr.AudioFile(wav_dosya_yolu) as source:
#             try:
#                 audio_data = recognizer.record(source)
#                 transkript = recognizer.recognize_google(audio_data, language="tr")
#             except sr.UnknownValueError:
#                 transkript = ""  # Tanınamayan değer hatası durumunda transkripti boş bir dize olarak ayarla

#         # Ses dosyasının transkripti ile dil tahmini yapma
#         kelime_listesi = wordpunct_tokenize(transkript)
#         turkce_kelime_sayisi = sum(1 for kelime in kelime_listesi if kelime.isalpha() and kelime.isascii() and kelime.lower() in turkce_stopwords)

#         # Ses dosyasının transkripti ile dil tahmini yapma
#         inputs = tokenizer(transkript, return_tensors="pt")
#         outputs = model(**inputs)
#         predicted_class = outputs.logits.argmax().item()

#         # Gerçek dil tahminini belirleme
#         real_language = "Turkish" if "tr" in mp4_dosyasi else "English"

#         language_labels = ["English", "Turkish"]
#         # Türkçe veya İngilizce dil tahminini ve transkripti yazdırma
#         if turkce_kelime_sayisi > 0:
#             predicted_language = "Turkish"
#         else:
#             predicted_language = "English"

#         print(f"{mp4_dosyasi} dosyasındaki gerçek dil: {real_language}")
#         print(f"{mp4_dosyasi} dosyasındaki dil tahmini: {predicted_language}")
#         print(f"Transkript: {transkript}")

#         # Kullanıcıdan bir tuşa basılmasını bekleyin
#         input(f"{mp4_dosyasi} dosyasını dinledikten sonra devam etmek için bir tuşa basın...")

# print("İşlem tamamlandı.")






