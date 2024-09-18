# Proje Sunum Web Sitesi

Bu proje, iki projenin sunumu için hazırlanmış bir web sitesidir. Web sitesi, Django framework kullanılarak geliştirilmiştir ve iki ana bölümden oluşur: MOBESE görüntü analizi ve ses tanıma. 

## Proje Açıklaması

Web sitesi, iki ana projeyi sunar:
1. **MOBESE Görüntü Analizi**: Proje kapsamında yapılan görseller ve açıklamalar bu bölümde yer alır.
2. **Ses Tanıma ve Duygu Durumu Analizi**: Yapay zeka modelleri entegre edilmiştir ve çalışabilir bir şekilde web sitesine gömülmüştür. Ses kısmı için basit bir arayüz tasarlanmıştır.

## Gereksinimler

- `Django`
- `numpy`
- `tensorflow` (veya `keras`)
- `librosa`
- `huggingface-transformers`

Bu kütüphaneleri yüklemek için:
```bash
pip install django numpy tensorflow librosa huggingface-transformers
```

Django uygulamasını başlatmak için konsola bunu yazmalısınız:
```bash
python manage.py runserver
```

## Web Site Çıktıları

Aşağıda, web sitesinin bazı örnek çıktıları yer almaktadır:

### Örnek 1

**Görüntü:**
![main_page](https://github.com/user-attachments/assets/8fcbe5b6-8d34-4e52-99f7-ce687bfe4ecc)


**Açıklama:** Ana ekran görüntüsü.

### Örnek 2

**Görüntü:**
![Screenshot_26](https://github.com/user-attachments/assets/7c4efe3b-6ee0-4e42-ad04-c5db3cd2250c)


**Açıklama:** Mobese kısmının küçük bir parçası.

### Örnek 3

**Görüntü:**
![voice](https://github.com/user-attachments/assets/df78782e-97dc-43b6-9ea9-fd9863b4fe49)


**Açıklama:** Ses yükleme için kullanılan basit form ekranı.

### Örnek 4

**Görüntü:**
![voice_output](https://github.com/user-attachments/assets/3979b933-54a3-4d87-8897-a29b17ee9e23)


**Açıklama:** Ses yükleme sonrası yapay zeka modellerinden dönen çıktıların HTML sayfasına yansıyan görüntüsü.
