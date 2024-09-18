from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic.base import TemplateView
from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from .forms import AudioFileForm
from .models import AudioFile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from django.conf import settings
import os
import cv2
from google.cloud import translate
from google.cloud import speech
from google.oauth2 import service_account
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import io

model_path = 'app/model/voice_recognation_model.keras'
model_recognation = tf.keras.models.load_model(model_path)
client_file = "app/services/speech_to_text_key.json"
credentials = service_account.Credentials.from_service_account_file(client_file)
client_speech = speech.SpeechClient(credentials=credentials)
translate_client = translate.TranslationServiceClient(credentials=credentials)
pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

def home_page_view(request):
    return render(request, 'main_page.html')

def voice_view(request):
    return render(request, "voice.html")

def upload_audio(request):
    if request.method == 'POST':
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect("/voice/output")
    else:
        form = AudioFileForm()
    return render(request, 'voice.html', {'form': form})

def save_mfccs(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    
    # Dosyayı kaydet
    mfccs_dir = os.path.join(settings.MEDIA_ROOT, 'mfccs')
    if not os.path.exists(mfccs_dir):
        os.makedirs(mfccs_dir)
    
    output_path = os.path.join(mfccs_dir, 'test_mfccs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Matplotlib figürünü kapatın
    return output_path

def handle_uploaded_file():
   
    test_image = cv2.imread("media/mfccs/test_mfccs.png")
    test_image = cv2.resize(test_image, (128, 128))
    test_image = np.expand_dims(test_image, axis=0)
    
    # Modeli kullanarak tahmin yapma
    prediction = model_recognation.predict(test_image)
    person_names = ['Yunus', 'Tuna', 'Kübra']
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_person_name = person_names[predicted_label]
    print(predicted_person_name)
    
    return predicted_person_name

def speech_to_text_tr(audio_file):

    with io.open(audio_file, 'rb') as f:
        content = f.read()
        audio = speech.RecognitionAudio(content = content)

    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.MP3, sample_rate_hertz = 44100, language_code = 'tr-TR')

    response = client_speech.recognize(config=config, audio=audio)

    for result in response.results:
        text = result.alternatives[0].transcript

    return text

def translate_text(text):

    client = translate.TranslationServiceClient()

    location = "global"

    project_id = "YOUR PROJECT ID"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = translate_client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "tr",
            "target_language_code": "en-US",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print(f"Translated text: {translation.translated_text}")

    return translation.translated_text

def detect_emotion(audio_file):

    text = speech_to_text_tr(audio_file)
    text = translate_text(text)
    results = pipe(text, return_all_scores=True)
    results = results[0]  # `return_all_scores=True` olduğundan ilk eleman liste oluyor

    # En yüksek puanlanan ilk 3 duygu kategorisini alma
    top_3_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    emotion_dict = {}
    # Sonuçları yazdırma
    print("En yüksek puanlanan 3 duygu:")
    for emotion in top_3_emotions:
        emotion_dict[emotion['label']]=emotion['score']
    return emotion_dict

def show_outputs(request):
    audio = AudioFile.objects.order_by('id').last()
    save_mfccs("media/audio_files/{}".format(audio.title+".opus"))
    kisi = handle_uploaded_file()
    emotion = detect_emotion("media/audio_files/{}".format(audio.title+".opus"))
    context = {'audio': audio, 'ses':kisi, 'duygu':emotion}
    return render(request, 'voice_output.html', context)

def mobese_view(request):
    images = [
        ('/media/images/exa.jpg', 'Bu görselde proje için kullanılan İtalya da bulunan Perugia şehrinin görseli bulunuyor.'),
        ('/media/images/gray_scaled.jpg', 'Bu görselin üzerinde filtreler kullanabilmek için görseli gray scale olarak adlandırılan renkleri siyah ve beyaz tonlar arasına dönüştürerek tüm resmi gri tonlarında görüntülemeyi sağlayan fonksiyonu kullandım.'),
        ('/media/images/edges.jpg', 'Bu görsel kenarları belirgin hale getirmek için kullanılan Canny fonksiyonun bir çıktısıdır. Bu çıktı GaussianBlur olarak adlandırılan gürültüleri azaltmak ve detayları yumuşatmak için kullanılan fonksiyon, gray scaled olmuş görselin üzerinde kullanılarak elde edilmiştir.'),
        ('/media/images/edges_blursuz.jpg', 'Bu görsel gray scaled olmuş olan görseli gaussianblur fonksiyonu kullanmadan nasıl gözüktüğünü göstermek için elde edilmiştir.'),
        ('/media/images/otsu_output.jpg', 'Bu görsel otsu filtresi kullanılarak elde edilmiştir. Gri tonda olan görüntünün renk histogramı hesaplanır. Görüntünün arka plan ve ön plan olarak 2 renk sınıfında olduğu varsayılır. Bu iki renk sınıfının varyans değerleri hesaplanır. Varyans değerini en küçük bulan değer optimum eşik değeridir. Bu değer otsu eşiği olarak adlandırılır. Bu yöntem otomatik eşikleme işlemi yapmak için kullanılır. Bu filtre sonrasında görüntü daha iyi bir şekilde bölünür ve uygulanacak olan filtreler daha doğru sonuçlar verebilir.'),
        ('/media/images/Contours.jpg', 'Bu görsel bulunan karakterlerin etrafına sınır çizgileri çekilmesiyle oluşturulmuştur.'),
        ('/media/images/dilated_output.jpg', 'Genişleme ve erozyon olarak bulunan morfoloji işlemlerinden genleşme yöntemini kullandım. Bu yöntem çizgilerin daha büyük olmasını sağlar.'),
        ('/media/images/tracked_left.jpg', 'Bu resimde bulunan karakterlere id ler eklenmiştir. Tespit listeleri oluşturulmuştur. Tespit listesi sort fonksiyonun içerisine alınmıştır. Böylece tracker kullanılmıştır. Ardından in ve out durumlarını kontrol etmek için bir önceki ve şuan işlemde olan tracker objesinin in ve out olma durumları karşılaştırılır. Bir farklılık var ise duruma göre çıkan sayısına veya giren sayısına 1 eklenir'),
        ('/media/images/toplam_insan.jpg', 'Bu resimde meydanda bulunan insanlar tespit edilmeye çalışılmıştır. Tespit edilen kişi sayısı görüntüde total adı altında tutulur. Önceki görselde anlatılan ekleme ve çıkarma işlemine göre total sayı güncellenir.'),
    ]   

    return render(request, 'mobese.html', {'images': images})

