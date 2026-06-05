Python 3D Topology Optimization
Motoru
Proje Hakkında
Bu proje, eklemeli imalat (additive manufacturing) süreçleri için optimize edilmiş, yüksek
mukavemetli ve hafif organik parçalar tasarlayan bir 3D topoloji optimizasyonu motorudur.
Özellikler
● SIMP, BESO ve LSM algoritmaları içerir.
● NVIDIA Ekran Kartı (GPU) ve CPU destekli hesaplama.
● STL ve DXF (3D CAD) formatlarında çıktı.
● Eklemeli imalat için 45° desteksiz büyüme filtresi.
● Kullanıcı dostu arayüz.
Kurulum ve Kullanım
1. Gerekli kütüphaneleri yükleyin: pip install numpy scipy scikit-image trimesh
opencv-python ezdxf
2. GPU kullanımı için CuPy yüklemeyi unutmayın.
3. python optimize_3d.py komutuyla programı başlatın.
4. Arayüz üzerinden parametrelerinizi ayarlayın ve "3D Organik Üretimi Başlat" butonuna
tıklayın.
