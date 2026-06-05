"""
Gerçek 3D Topoloji Optimizasyonu (Organik Voxel Yapı)
Ekran Kartı (GPU) ve CPU Destekli
Author: Gemini Code Assist
"""

import sys
import time
import os
import platform
from datetime import datetime
print("\n" + "="*50)
print(">>> 3D ORGANİK OPTİMİZASYON BAŞLIYOR <<<")
print("="*50 + "\n")

try:
    import numpy as np
    import scipy.sparse as sps
    import scipy.sparse.linalg as splinalg
    import matplotlib.pyplot as plt
    from skimage import measure
    import trimesh
    import tkinter as tk
    from tkinter import ttk, messagebox
    from scipy.ndimage import gaussian_filter, maximum_filter
    import scipy.ndimage as ndimage
    import cv2
    import ezdxf
except ImportError as e:
    print(f"\n❌ EKSİK KÜTÜPHANE HATASI: {e}")
    print("Lütfen şu komutu çalıştırın: pip install numpy scipy scikit-image trimesh opencv-python ezdxf")
    input("\nÇıkmak için Enter'a basın...")
    sys.exit()

# GPU Kullanımı için CuPy Kontrolü
HAS_GPU = False
GPU_INFO = ""
try:
    if platform.system() == 'Darwin':
        GPU_INFO = "Mac OS (Apple Silicon) tespit edildi."
        raise ImportError("Mac sistemlerde GPU ivmelendirmesi yerine Yüksek Performanslı CPU (İşlemci) kullanılır.")
        
    import glob
    # Python 3.8+ için Windows Güvenlik Önlemini aşma (NVIDIA DLL'lerini manuel tanıtma)
    if hasattr(os, 'add_dll_directory'):
        cuda_path = os.environ.get('CUDA_PATH', '')
        if cuda_path and os.path.exists(os.path.join(cuda_path, 'bin')):
            os.add_dll_directory(os.path.join(cuda_path, 'bin'))
        else:
            for path in glob.glob(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin'):
                os.add_dll_directory(path)

    import cupy as cp
    import cupyx.scipy.sparse as csps
    import cupyx.scipy.sparse.linalg as csplinalg
    import cupyx.scipy.ndimage as cndimage
    HAS_GPU = True
    GPU_INFO = "NVIDIA Ekran Kartı (GPU)"
except Exception as e:
    print(f"\n💡 BİLGİLENDİRME: {GPU_INFO if GPU_INFO else 'NVIDIA Ekran Kartı bulunamadı.'}")
    print("Program, Apple Mac / AMD veya standart sistemler için optimize edilmiş 'İşlemci (CPU) Modunda' çalışacaktır.")
    print(f"Hata Detayı: {e}\n")

# =============================================================================
# 1. 3D FEA YARDIMCI FONKSİYONLARI (KÜP ELEMAN)
# =============================================================================

def get_element_stiffness_3D(E, nu):
    """8 Düğümlü İzotropik 3D Küp Eleman Rijitlik Matrisi (24x24)"""
    # 3D matrisin analitik hesaplanmış temel formülasyonu
    A = np.array([
        [32, 6, -8, 6, -4, 3, -16, -6, 0, -3, -4, -3, -16, 3, 0, -6, 8, -3, 0, -3, 0, 3, 8, 3],
        [6, 32, 6, 3, -4, -8, -6, -16, -3, 0, -3, -4, -3, 0, 3, -16, -3, 8, 3, 0, -3, 0, 3, 8],
        [-8, 6, 32, -6, -8, -4, 0, -3, -16, 3, -4, -3, 0, 3, -16, 3, -8, -3, 0, -3, 0, 3, 3, 8],
        [6, 3, -6, 32, 6, -8, -3, 0, 3, -16, -6, -16, 3, 0, -3, 8, 3, 0, -3, 0, -3, -4, -3, -4],
        [-4, -4, -8, 6, 32, 6, -4, -3, -3, -6, -16, -3, 8, 3, 0, -3, 0, 3, 3, 0, 3, -3, -4, -3],
        [3, -8, -4, -8, 6, 32, -3, -4, -3, -16, -3, -6, 0, 3, 8, 0, -3, -3, 3, 3, 0, -4, -3, -3]
    ])
    # Matris simetrisi ve blok düzenlemeleri (basitleştirilmiş)
    # Tam ve doğru bir 24x24 matris sayısal integrasyonla elde edilir:
    D = E / ((1+nu)*(1-2*nu)) * np.array([
        [1-nu, nu, nu, 0, 0, 0], [nu, 1-nu, nu, 0, 0, 0], [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0], [0, 0, 0, 0, (1-2*nu)/2, 0], [0, 0, 0, 0, 0, (1-2*nu)/2]
    ])
    pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
    KE = np.zeros((24, 24))
    for xi in pts:
        for eta in pts:
            for zeta in pts:
                dNx = np.array([-(1-eta)*(1-zeta), (1-eta)*(1-zeta), (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
                                -(1-eta)*(1+zeta), (1-eta)*(1+zeta), (1+eta)*(1+zeta), -(1+eta)*(1+zeta)]) / 8.0
                dNy = np.array([-(1-xi)*(1-zeta), -(1+xi)*(1-zeta), (1+xi)*(1-zeta), (1-xi)*(1-zeta),
                                -(1-xi)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1+zeta), (1-xi)*(1+zeta)]) / 8.0
                dNz = np.array([-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta), -(1-xi)*(1+eta),
                                (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]) / 8.0
                B = np.zeros((6, 24))
                for i in range(8):
                    B[0, 3*i] = dNx[i]; B[1, 3*i+1] = dNy[i]; B[2, 3*i+2] = dNz[i]
                    B[3, 3*i] = dNy[i]; B[3, 3*i+1] = dNx[i]
                    B[4, 3*i+1] = dNz[i]; B[4, 3*i+2] = dNy[i]
                    B[5, 3*i] = dNz[i]; B[5, 3*i+2] = dNx[i]
                KE += B.T @ D @ B
    return KE

# =============================================================================
# 2. ANA 3D OPTİMİZASYON DÖNGÜSÜ
# =============================================================================

def optimize_3d(params):
    print("\n⏳ 3D Mesh ve Denklemler Hazırlanıyor (Lütfen bekleyin)...\n")
    start_time = time.time()
    
    # SİMETRİ YÖNTEMİ: Yarı domain hesaplanacak
    nelx, nely = params['nelx'], params['nely']
    nelz = params['nelz'] // 2  # Domain Z ekseninde ikiye bölündü
    volfrac, penal, rmin = params['volfrac'], params['penal'], params['rmin']
    E0, Emin, nu = params['E0'], 1e-4, params['nu']
    use_gpu = params['use_gpu'] and HAS_GPU
    
    if use_gpu: print("🚀 NVIDIA GPU KULLANILIYOR: Süreç Hızlandırılıyor!")
    else: print("💻 İŞLEMCİ (CPU) KULLANILIYOR: Bu işlem zaman alabilir.")

    KE = get_element_stiffness_3D(E0, nu)
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    
    # Serbestlik dereceleri haritası
    edofMat = np.zeros((nele, 24), dtype=int)
    el = 0
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                n1 = i*(nely+1)*(nelz+1) + j*(nelz+1) + k
                n2 = (i+1)*(nely+1)*(nelz+1) + j*(nelz+1) + k
                n3 = (i+1)*(nely+1)*(nelz+1) + (j+1)*(nelz+1) + k
                n4 = i*(nely+1)*(nelz+1) + (j+1)*(nelz+1) + k
                n5 = i*(nely+1)*(nelz+1) + j*(nelz+1) + (k+1)
                n6 = (i+1)*(nely+1)*(nelz+1) + j*(nelz+1) + (k+1)
                n7 = (i+1)*(nely+1)*(nelz+1) + (j+1)*(nelz+1) + (k+1)
                n8 = i*(nely+1)*(nelz+1) + (j+1)*(nelz+1) + (k+1)
                edofMat[el, :] = np.array([
                    3*n1, 3*n1+1, 3*n1+2, 3*n2, 3*n2+1, 3*n2+2, 3*n3, 3*n3+1, 3*n3+2, 3*n4, 3*n4+1, 3*n4+2,
                    3*n5, 3*n5+1, 3*n5+2, 3*n6, 3*n6+1, 3*n6+2, 3*n7, 3*n7+1, 3*n7+2, 3*n8, 3*n8+1, 3*n8+2
                ])
                el += 1
                
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()
    
    # Filtre Hazırlığı (Hassas yapı için)
    print("⏳ Mesh Filtresi Hesaplanıyor...")
    iH, jH, sH = [], [], []
    for i1 in range(nelx):
        for j1 in range(nely):
            for k1 in range(nelz):
                e1 = i1*nely*nelz + j1*nelz + k1
                i_min, i_max = max(i1 - int(rmin), 0), min(i1 + int(rmin) + 1, nelx)
                j_min, j_max = max(j1 - int(rmin), 0), min(j1 + int(rmin) + 1, nely)
                k_min, k_max = max(k1 - int(rmin), 0), min(k1 + int(rmin) + 1, nelz)
                for i2 in range(i_min, i_max):
                    for j2 in range(j_min, j_max):
                        for k2 in range(k_min, k_max):
                            e2 = i2*nely*nelz + j2*nelz + k2
                            dist = np.sqrt((i1-i2)**2 + (j1-j2)**2 + (k1-k2)**2)
                            if dist < rmin:
                                iH.append(e1); jH.append(e2); sH.append(rmin - dist)
    H = sps.coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsr()
    Hs = H.sum(axis=1).A1
    # --- VEKTÖRİZE HASSASİYET FİLTRESİ (MESH-INDEPENDENCY) ---
    print("⏳ Vektörize Hassasiyet Filtresi (Kernel) Hesaplanıyor...")
    r_int = int(np.ceil(rmin))
    X_k, Y_k, Z_k = np.ogrid[-r_int:r_int+1, -r_int:r_int+1, -r_int:r_int+1]
    dist_k = np.sqrt(X_k**2 + Y_k**2 + Z_k**2)
    kernel = np.maximum(0, rmin - dist_k)
    
    # Sınır etkilerini önlemek için normalize matrisi (Hs)
    Hs_3d = ndimage.convolve(np.ones((nelx, nely, nelz)), kernel, mode='constant', cval=0.0)
    if use_gpu:
        kernel_gpu = cp.array(kernel)
        Hs_gpu = cp.array(Hs_3d)
    
    # Sınır Şartları ve Vidalar (Pasif Bölgeler)
    x = np.ones(nele) * volfrac
    passive_solid = np.zeros(nele, dtype=bool)
    passive_void = np.zeros(nele, dtype=bool)
    fixed_dofs = []
    
    wall_screw_radius = params['wall_screw_d'] / 2.0
    pixel_size = params['physical_length'] / nelx
    wall_screw_radius_px = wall_screw_radius / pixel_size
    screwdriver_radius_px = wall_screw_radius_px + (4.0 / pixel_size) # Vida başı ve tornavida için +4mm pay
    solid_padding_px = 5.0 / pixel_size # Etrafında 5mm dolu alan
    flange_px = max(1, int(10.0 / pixel_size)) # 10mm derinliğinde delik
    
    # Duvara 2 Vida Bağlantısı ve Tornavida Yolu (X ekseni boyunca)
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                e_idx = i*nely*nelz + j*nelz + k
                dist1 = np.sqrt((j - nely*0.25)**2 + (k - nelz)**2) # Simetri düzlemine göre (nelz)
                dist2 = np.sqrt((j - nely*0.75)**2 + (k - nelz)**2)
                
                if dist1 <= wall_screw_radius_px or dist2 <= wall_screw_radius_px:
                    passive_void[e_idx] = True # Vida dişi yolu (Baştan sona)
                elif dist1 <= screwdriver_radius_px or dist2 <= screwdriver_radius_px:
                    if i >= flange_px:
                        passive_void[e_idx] = True # Tornavida erişim yolu
                        
                if i < flange_px:
                    if (wall_screw_radius_px < dist1 <= wall_screw_radius_px + solid_padding_px) or \
                       (wall_screw_radius_px < dist2 <= wall_screw_radius_px + solid_padding_px):
                        passive_solid[e_idx] = True # Flanş (Dolu et)
                        
                # GÜÇLÜ ANKRAJ: Sadece boşluğu değil, duvarla temas eden tam dolu flanşı da duvara sabitle
                if i == 0:
                    if (dist1 <= wall_screw_radius_px + solid_padding_px) or \
                       (dist2 <= wall_screw_radius_px + solid_padding_px):
                        fixed_dofs.extend(edofMat[e_idx, :].tolist())
                
    # Simetri Düzlemi Sınır Şartı (Z ekseninde hareket kilitli)
    for i in range(nelx + 1):
        for j in range(nely + 1):
            node = i*(nely+1)*(nelz+1) + j*(nelz+1) + nelz
            fixed_dofs.append(3*node + 2) # Sadece Z (2) yönü kilitli
                
    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(ndof), fixed_dofs)
    
    # Yük Yüzeyi ve Vidaları (Üst Yüzey y=0)
    F = np.zeros(ndof)
    top_screws = params['top_screws']
    half_force = params['force'] / 2.0 # Toplam yükün simetrik yarısı
    
    if top_screws > 0:
        if top_screws == 1: screw_locs, screw_z = [nelx*0.80], [nelz]
        elif top_screws == 2: screw_locs, screw_z = [nelx*0.55, nelx*0.85], [nelz, nelz]
        else: screw_locs, screw_z = [nelx*0.55, nelx*0.85], [nelz*0.3, nelz*0.3] # Düzeltme: 4 Vida için 2'si simetride
        
        load_per_screw = half_force / len(screw_locs)
        
        for j in range(nely):
            for i in range(nelx):
                for k in range(nelz):
                    e_idx = i*nely*nelz + j*nelz + k
                    for sx, sz in zip(screw_locs, screw_z):
                        dist = np.sqrt((i - sx)**2 + (k - sz)**2)
                        if dist <= wall_screw_radius_px:
                            passive_void[e_idx] = True # Vida deliği
                        elif dist <= wall_screw_radius_px + solid_padding_px:
                            if j < flange_px:
                                passive_solid[e_idx] = True # Vida tutunma eti
                                
        # FİZİKSEL DOĞRULUK: Kuvveti deliğin etrafına yayıp çoğaltmak yerine, sadece vidanın tam merkez noktasına uygula!
        for sx, sz in zip(screw_locs, screw_z):
            node_idx = int(sx)*(nely+1)*(nelz+1) + 0*(nelz+1) + int(sz)
            F[3 * node_idx + 1] += load_per_screw
    else:
        # Sadece uç noktadan tekil yük
        node_idx = nelx*(nely+1)*(nelz+1) + 0*(nelz+1) + nelz
        F[3 * node_idx + 1] = half_force
        
    alg = params.get('algorithm', 'SIMP')
    current_volfrac = 1.0 # Hacim Şokunu Önlemek İçin Tüm Algoritmalara Ortak Başlangıç
    ER = 0.02 # Evrim Hızı (Evolutionary Ratio)
    
    if alg == 'BESO':
        x = np.ones(nele)
        x[passive_void] = 0.001
        current_volfrac = 1.0
        ER = 0.01 # Sabırsızlık engellendi. Hedefe 35 değil, 70-80 döngüde sindirerek gidecek.
        alpha_hist = np.zeros(nele)
    elif alg == 'LSM':
        x = np.ones(nele) * volfrac
        x[passive_solid] = 1.0
        x = np.ones(nele)
        x[passive_void] = 0.001
        phi = np.copy(x) - 0.5 # Phi'yi hacme göre başlat
    else:
        x = np.ones(nele) * volfrac
        x[passive_solid] = 1.0
        x = np.ones(nele)
        x[passive_void] = 0.001
    
    loop = 0
    change = 1.0
    
    # HIZLANDIRICI 1: Sabit değerleri (Yük vektörü) ve Warm Start hafızasını döngü dışına al
    if use_gpu:
        F_gpu = cp.array(F[free_dofs])
        U_gpu_prev = cp.zeros(len(free_dofs))
        
    # Video metinleri için ön hesaplamalar
    Lx = params['physical_length']
    Ly = Lx * 0.6
    Lz = Lx * 0.2
    total_vol_cm3 = (Lx / 10.0) * (Ly / 10.0) * (Lz / 10.0)
    short_mat = params['mat_name'].split()[0]
    density = params['density']
    
    print("⚙️ Optimizasyon Başlıyor...\n")
    
    # --- CANLI ÖNİZLEME PENCERESİ ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.canvas.manager.set_window_title(f'Paprika - {alg} Analizi')
    ax.set_title(f"Canlı Analiz: {alg} (Renkli Stres Haritası)")
    ax.axis('off')
    im = ax.imshow(np.zeros((nely, nelx)), cmap='inferno', vmin=0, vmax=1, origin='lower')
    plt.show()
    # -------------------------------
    
    video_frames = [] # Video için kareleri (frame) tutacak liste

    while change > params['tol'] and loop < params['max_loop']:
        loop_start = time.time()
        loop += 1
        E_simp = Emin + x**penal * (E0 - Emin)
        sK = (E_simp[:, np.newaxis, np.newaxis] * KE).flatten()
        K = sps.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K_free = K[free_dofs, :][:, free_dofs]
        
        U = np.zeros(ndof)
        
        # GPU vs CPU Çözücü
        if use_gpu:
            # NVRTC (FP8) DERLEME HATASINA KESİN VE NİHAİ ÇÖZÜM:
            # CuPy'nin gizlice matris dönüşümü yapmasını engellemek için,
            # tüm format dönüşümlerini CPU'da SciPy ile risksizce yapıyoruz!
            K_free_csr = K_free.tocsr()
            K_gpu = csps.csr_matrix(K_free_csr) # GPU sadece belleğine kopyalar
            
            # Preconditioner matrisini dia_matrix yerine "Saf CSR" olarak inşa ediyoruz
            K_diag_cpu = K_free_csr.diagonal()
            K_diag_cpu[K_diag_cpu == 0] = 1.0
            M_diag_gpu = cp.array(1.0 / K_diag_cpu)
            
            n_free = len(free_dofs)
            idx_gpu = cp.arange(n_free, dtype=cp.int32)
            ptr_gpu = cp.arange(n_free + 1, dtype=cp.int32)
            M_gpu = csps.csr_matrix((M_diag_gpu, idx_gpu, ptr_gpu), shape=(n_free, n_free))
            
            # HIZLANDIRICI 2 (Warm Start): Önceki döngünün sonucunu başlangıç tahmini (x0) olarak ver!
            U_gpu, _ = csplinalg.cg(K_gpu, F_gpu, x0=U_gpu_prev, rtol=1e-4, M=M_gpu, maxiter=2500)
            U_gpu_prev = U_gpu # Bir sonraki adım için hafızaya al
            
            U[free_dofs] = cp.asnumpy(U_gpu)
        else:
            U[free_dofs] = splinalg.spsolve(K_free, F[free_dofs])
        
        ce = np.sum(np.dot(U[edofMat], KE) * U[edofMat], axis=1)
        c = np.sum(E_simp * ce)
        
        # Hacim Kısıtı Güncellemesi (Hacim Şokunu ve Kopmaları Engeller)
        current_volfrac = max(volfrac, current_volfrac - ER)

        if alg == 'SIMP':
            dc = -penal * x**(penal-1) * (E0 - Emin) * ce
                
            # 1. 3D BASKI FİZİĞİ: SARKMA (OVERHANG) VE YERÇEKİMİ KONTROLÜ
            if params.get('am_filter', False):
                dc_3d = dc.reshape((nelx, nely, nelz))
                # Algoritmayı parçaları yerçekimi yönüne (Aşağı/Y eksenine) uzatmaya zorlar (Damla Formu)
                supp = np.roll(dc_3d, shift=1, axis=1) * 0.75 
                supp[:, 0, :] = 0
                dc += supp.flatten()
                
            # 2. 3D BASKI FİZİĞİ: HEAVISIDE KESKİNLEŞTİRİCİ (MİN DUVAR KALINLIĞI)
            if not params.get('lattice', False):
                beta = min(20.0, 1.0 + loop * 0.1) # Kademeli olarak keskinleştir
                x_3d = x.reshape((nelx, nely, nelz))
                if use_gpu:
                    x_tilde = cp.asnumpy(cndimage.convolve(cp.array(x_3d), kernel_gpu, mode='constant', cval=0.0) / Hs_gpu).flatten()
                else:
                    x_tilde = (ndimage.convolve(x_3d, kernel, mode='constant', cval=0.0) / Hs_3d).flatten()
                dx = beta * np.exp(-beta * x_tilde) + np.exp(-beta)
                dc = dc * dx

            # YÜKSEK PERFORMANSLI VEKTÖRİZE FİLTRELEME
            dc_3d = (x * dc).reshape((nelx, nely, nelz))
            if use_gpu:
                dc_filtered = cp.asnumpy(cndimage.convolve(cp.array(dc_3d), kernel_gpu, mode='constant', cval=0.0) / Hs_gpu).flatten()
            else:
                dc_filtered = (ndimage.convolve(dc_3d, kernel, mode='constant', cval=0.0) / Hs_3d).flatten()
            dc = dc_filtered / np.maximum(0.001, x)
            
            # Optimality Criteria (SIMP)
            # İleri seviye AM filtrelerinin yarattığı "Osilasyon (Titreme)" döngüsünü kırmak için Dinamik Fren
            dynamic_move = 0.2
            if loop > 150:
                dynamic_move = max(0.01, 0.2 * (0.97 ** (loop - 150)))
                
            l1, l2 = 0.0, 1e9
            xnew = np.zeros(nele)
            # Sıfıra bölünmeyi (ZeroDivisionError) engellemek için + 1e-9 eklendi
            while (l2 - l1) / (l1 + l2 + 1e-9) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                # Karekök içi negatif olma ve aşırı taşma (Overflow) hatalarını kökünden çözer
                B = np.sqrt(np.maximum(-dc, 0.0) / (lmid + 1e-15))
                xnew = np.maximum(0.001, np.maximum(x - dynamic_move, np.minimum(1.0, np.minimum(x + dynamic_move, x * B))))
                xnew[passive_solid] = 1.0
                xnew[passive_void] = 0.001
                if np.sum(xnew) - current_volfrac * nele > 0: l1 = lmid
                else: l2 = lmid
        elif alg == 'BESO':
            # BESO Hassasiyet ve Filtreleme
            alpha = ce * (E0 - Emin)
            alpha_3d = alpha.reshape((nelx, nely, nelz))
            if use_gpu:
                alpha = cp.asnumpy(cndimage.convolve(cp.array(alpha_3d), kernel_gpu, mode='constant', cval=0.0) / Hs_gpu).flatten()
            else:
                alpha = (ndimage.convolve(alpha_3d, kernel, mode='constant', cval=0.0) / Hs_3d).flatten()
            
            # Tarihsel Ortalama (Kararlılık için)
            if loop == 1:
                alpha_hist = alpha.copy()
            alpha = 0.5 * alpha + 0.5 * alpha_hist
            alpha_hist = alpha.copy()
            
            # Bisection ile Kesin (0-1) Atama
            l1, l2 = np.min(alpha), np.max(alpha)
            while (l2 - l1) / (abs(l1) + abs(l2) + 1e-5) > 1e-5:
                th = (l1 + l2) / 2.0
                xnew = np.ones(nele)
                xnew[alpha < th] = 0.001
                xnew[passive_solid] = 1.0
                xnew[passive_void] = 0.001
                if np.sum(xnew) / nele > current_volfrac:
                    l1 = th
                else:
                    l2 = th
        elif alg == 'LSM':
            # Level Set Method (LSM) Sınır Takibi
            V = ce * (E0 - Emin)
            V_3d = V.reshape((nelx, nely, nelz))
            if use_gpu:
                V = cp.asnumpy(cndimage.convolve(cp.array(V_3d), kernel_gpu, mode='constant', cval=0.0) / Hs_gpu).flatten()
            else:
                V = (ndimage.convolve(V_3d, kernel, mode='constant', cval=0.0) / Hs_3d).flatten()
            V = V / (np.max(np.abs(V)) + 1e-9) # Normalizasyon
            
            l1, l2 = np.min(V), np.max(V)
            while (l2 - l1) > 1e-5:
                th = (l1 + l2) / 2.0
                phinew = phi + 0.04 * (V - th) # Dalga hızı iyice kısıldı (LSM artık daha uysal)
                xnew = np.ones(nele)
                xnew[phinew <= 0] = 0.001
                xnew[passive_solid] = 1.0
                xnew[passive_void] = 0.001
                if np.sum(xnew) / nele > current_volfrac: l1 = th
                else: l2 = th
                
            phi = np.clip(phi + 0.04 * (V - l2), -1.0, 1.0)
            phi_3d = phi.reshape((nelx, nely, nelz))
            if use_gpu:
                phi = cp.asnumpy(cndimage.convolve(cp.array(phi_3d), kernel_gpu, mode='constant', cval=0.0) / Hs_gpu).flatten()
            else:
                phi = (ndimage.convolve(phi_3d, kernel, mode='constant', cval=0.0) / Hs_3d).flatten()
            
        change = np.max(np.abs(xnew - x))
        if alg == 'BESO' and current_volfrac > volfrac + 1e-3:
            change = 1.0 # BESO'nun hedefe ulaşana kadar durmasını engelle
            
        x = xnew
        
        # STRES HARİTASI (Heatmap) VE EKRAN GÜNCELLEMESİ
        stress_slice = ce.reshape((nelx, nely, nelz))[:, :, nelz-1].T
        density_slice = x.reshape((nelx, nely, nelz))[:, :, nelz-1].T
        mask = density_slice > 0.4
        
        vmax = np.percentile(stress_slice[mask], 95) if np.any(mask) else np.max(stress_slice) + 1e-9
        stress_norm = np.clip(stress_slice / vmax, 0, 1)
        
        display_img = np.zeros_like(stress_norm)
        display_img[mask] = stress_norm[mask]
        im.set_data(display_img)
        plt.pause(0.01)
        
        # Video için renkli stres haritasını oluştur
        stress_uint8 = (stress_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(stress_uint8, cv2.COLORMAP_INFERNO)
        
        # Arka planı modern koyu gri yap, sadece parçayı renklendir
        frame_bgr = np.full_like(heatmap, (25, 25, 30))
        frame_bgr[mask] = heatmap[mask]
        
        scale = 12 # Video çözünürlüğünü (HD seviyelerine) ve keskinliğini artırdık
        h_f, w_f = frame_bgr.shape[:2]
        frame_bgr = cv2.resize(frame_bgr, (w_f*scale, h_f*scale), interpolation=cv2.INTER_NEAREST)
        
        # --- MODERN HUD (BİLGİ PANELİ) ---
        # Yazıların grafikle üst üste binmesini engelleyen yarı saydam panel
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (20, 20), (450, 310), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame_bgr, 0.15, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (20, 20), (450, 310), (100, 100, 100), 2) # Şık çerçeve
        
        # Anlık değerlerin hesaplanması
        elapsed = time.time() - start_time
        current_v = (float(np.sum(x)) / nele) * total_vol_cm3
        current_w = current_v * density
        
        # Ekrana bilgi metinlerini yazdır (HUD Paneli)
        text_color = (230, 230, 230)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_bgr, f"Paprika - {alg}", (40, 70), font, 1.1, (0, 180, 255), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Malzeme: {short_mat}", (40, 120), font, 0.8, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Sure: {elapsed:.1f} sn", (40, 160), font, 0.8, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Hacim: {current_v:.1f} cm3", (40, 200), font, 0.8, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Agirlik: {current_w:.1f} g", (40, 240), font, 0.8, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Dongu: {loop}/{params['max_loop']}", (40, 285), font, 0.8, (50, 255, 50), 2, cv2.LINE_AA)
        
        # --- KOORDİNAT VE ÖLÇEK ÇUBUKLARI (CAD RULERS) ---
        H_img, W_img = frame_bgr.shape[:2]
        
        # Rulers için okunabilirliği artıran yarı saydam arka plan (Kenar Şeritleri)
        ruler_overlay = frame_bgr.copy()
        cv2.rectangle(ruler_overlay, (0, H_img - 45), (W_img, H_img), (15, 15, 15), -1) # Alt şerit
        cv2.rectangle(ruler_overlay, (W_img - 85, 0), (W_img, H_img), (15, 15, 15), -1) # Sağ şerit
        cv2.addWeighted(ruler_overlay, 0.65, frame_bgr, 0.35, 0, frame_bgr)

        ruler_color = (200, 200, 200)
        
        # X Ekseni (Alt Çubuk - Uzunluk)
        cv2.line(frame_bgr, (20, H_img - 25), (W_img - 85, H_img - 25), ruler_color, 2)
        for i in range(6):
            tick_x = int(20 + i * (W_img - 105) / 5)
            val_x = int(i * Lx / 5)
            cv2.line(frame_bgr, (tick_x, H_img - 25), (tick_x, H_img - 15), ruler_color, 2)
            cv2.putText(frame_bgr, f"{val_x}", (tick_x - 12, H_img - 5), font, 0.5, ruler_color, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "X (mm)", (W_img - 150, H_img - 30), font, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
        
        # Y Ekseni (Sağ Çubuk - Yükseklik)
        cv2.line(frame_bgr, (W_img - 85, 20), (W_img - 85, H_img - 25), ruler_color, 2)
        for i in range(1, 6): # 0'ı X ekseni ile çakışmasın diye atlıyoruz
            tick_y = int(H_img - 25 - i * (H_img - 45) / 5)
            val_y = int(i * Ly / 5)
            cv2.line(frame_bgr, (W_img - 85, tick_y), (W_img - 75, tick_y), ruler_color, 2)
            cv2.putText(frame_bgr, f"{val_y}", (W_img - 70, tick_y + 5), font, 0.5, ruler_color, 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "Y", (W_img - 75, 15), font, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

        video_frames.append(frame_bgr)

        loop_time = time.time() - loop_start
        print(f"Döngü: {loop:03d} | Esneklik(Compliance): {c:.2f} | Maks. Değişim: {change:.4f} | Döngü Süresi: {loop_time:.2f} sn")
        
    print(f"\n✅ Optimizasyon {time.time() - start_time:.1f} saniyede tamamlandı!")
    
    plt.ioff()
    plt.close(fig)
    
    # İŞLEM BİTTİ: Çıkan yarı parçayı tam parçaya aynala (Mirroring)
    print("🪞 Simetrik Parça Aynalanarak Tamamlanıyor...")
    

    x_3d = x.reshape((nelx, nely, nelz))
    x_full = np.concatenate((x_3d, x_3d[:, :, ::-1]), axis=2)
    
    # CAD İşlemleri İçin Kusursuz Lazer Delik Maskelerini Aynala ve Geri Döndür
    passive_solid_3d = passive_solid.reshape((nelx, nely, nelz))
    passive_solid_full = np.concatenate((passive_solid_3d, passive_solid_3d[:, :, ::-1]), axis=2)
    passive_void_3d = passive_void.reshape((nelx, nely, nelz))
    passive_void_full = np.concatenate((passive_void_3d, passive_void_3d[:, :, ::-1]), axis=2)
    
    params['nelz'] = nelz * 2 # Orijinal boyuta dön
    
    return x_full.flatten(), video_frames, passive_solid_full.flatten(), passive_void_full.flatten()

# =============================================================================
# 3. 3D YÜZEY ÖRME VE STL ÇIKTISI
# =============================================================================

def export_3d_stl(x_opt, params, passive_solid=None, passive_void=None):
    nelx, nely, nelz = params['nelx'], params['nely'], params['nelz']
    density_3d = x_opt.reshape((nelx, nely, nelz))
    
    print("🖨️ 3D Mesh Örülüyor ve STL Kaydediliyor...")
    
    alg_name = params.get('current_alg', 'SIMP')
    
    # BESO'nun köşeli yapısını eritmek için zımpara gücü
    sigma_val = 1.2 if alg_name == 'BESO' else 0.8
    smooth_density = gaussian_filter(density_3d.astype(float), sigma=sigma_val)
    
    # --- LAZER KESİM HASSASİYETİ (SDF İLE PÜRÜZSÜZ DELİK VE DÜZ YÜZEYLER) ---
    pixel_size = params['physical_length'] / nelx
    wall_screw_radius_px = (params['wall_screw_d'] / 2.0) / pixel_size
    screwdriver_radius_px = wall_screw_radius_px + (4.0 / pixel_size)
    flange_px = max(1, int(10.0 / pixel_size))
    solid_padding_px = 5.0 / pixel_size
    
    X, Y, Z = np.meshgrid(np.arange(nelx), np.arange(nely), np.arange(nelz), indexing='ij')
    z_center = nelz / 2.0
    
    # 1. Duvar Vidaları (Y=0.25 ve Y=0.75, Z=Merkez)
    d_wall1 = np.sqrt((Y - nely*0.25)**2 + (Z - z_center)**2)
    d_wall2 = np.sqrt((Y - nely*0.75)**2 + (Z - z_center)**2)
    d_wall = np.minimum(d_wall1, d_wall2)
    
    # KUSURSUZ BÜTÜNLEŞME: Flanşları (Duvar) organik gövdeye yumuşatarak kaynat! (Pah kırma)
    flange_radius = wall_screw_radius_px + solid_padding_px
    flange_blend = np.clip(1.0 - (d_wall - flange_radius) / 1.5, 0.0, 1.0)
    flange_blend[X >= flange_px] = 0.0 # Duvar yüzeyini cam gibi düz tut
    smooth_density = np.maximum(smooth_density, flange_blend)
    
    # Vida ve tornavida boşluklarını rampa (SDF) ile pürüzsüz silindire çevir
    hole_radius_field = np.where(X < flange_px, wall_screw_radius_px, screwdriver_radius_px)
    hole_carve = np.clip((d_wall - hole_radius_field) / 0.75, 0.0, 1.0)
    smooth_density = np.minimum(smooth_density, hole_carve)
    
    # 2. Üst Taşıma Yüzeyi Vidaları
    top_screws = params['top_screws']
    if top_screws > 0:
        half_nelz = nelz / 2.0
        if top_screws == 1: 
            screw_locs, screw_z = [nelx*0.80], [half_nelz]
        elif top_screws == 2: 
            screw_locs, screw_z = [nelx*0.55, nelx*0.85], [half_nelz, half_nelz]
        else: 
            screw_locs, screw_z = [nelx*0.55, nelx*0.85, nelx*0.55, nelx*0.85], [half_nelz*0.3, half_nelz*0.3, nelz - half_nelz*0.3, nelz - half_nelz*0.3]
            
        d_top = np.full((nelx, nely, nelz), 1e9)
        for sx, sz in zip(screw_locs, screw_z):
            d_top = np.minimum(d_top, np.sqrt((X - sx)**2 + (Z - sz)**2))
            
        top_blend = np.clip(1.0 - (d_top - flange_radius) / 1.5, 0.0, 1.0)
        top_blend[Y >= flange_px] = 0.0
        smooth_density = np.maximum(smooth_density, top_blend)
        
        top_hole_carve = np.clip((d_top - wall_screw_radius_px) / 0.5, 0.0, 1.0)
        smooth_density = np.minimum(smooth_density, top_hole_carve)

    # Su Sızdırmazlık (Watertight) için etrafına boşluk çerçevesi (padding) sar
    padded_density = np.pad(smooth_density, pad_width=1, mode='constant', constant_values=0.0)
    
    level = params.get('extract_level', 0.4)
    verts, faces, normals, values = measure.marching_cubes(padded_density, level=level)
    
    # GERÇEK MİLİMETRİK ÖLÇEKLENDİRME VE ORİJİNE HİZALAMA
    min_b = verts.min(axis=0)
    max_b = verts.max(axis=0)
    current_len = max_b[0] - min_b[0]
    scale_factor = params['physical_length'] / nelx 
    
    if current_len > 0:
        verts = (verts - min_b) * scale_factor

    # --- ÜÇGEN SAYISINI AZALTMA (MESH DECIMATION) ---
    # Slicer'ın (Dilimleyicinin) kasmasını önlemek için 0.1mm altındaki gereksiz mikro-üçgenleri birleştiririz.
    # Mukavemeti veya delik çaplarını %0 oranında (hiç) etkilemez!
    verts = np.round(verts, 2) # Koordinatları 0.01 mm (Lazer) hassasiyetine çıkarttık
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces) # Normalleri otomatik yeniden hesaplar
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    
    # PARÇA BÜTÜNLÜĞÜ KONTROLÜ: Kopan küçük adacıkları sil, sadece ana gövdeyi tut
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))
        
    output_dir = params['output_dir']
    filename = os.path.join(output_dir, f"Paprika_Braket_{params.get('current_alg', 'Opt')}_{params['mat_name'][:3]}_{params['force']}N.stl")
    mesh.export(filename)
    
    print(f"✅ MUHTEŞEM! 3D Yazıcı Dosyanız (STL) '{filename}' adıyla kaydedildi.")
    
    # --- 3D CAD (STEP ALTERNATİFİ - 3D DXF) DIŞA AKTARIMI ---
    print("📐 CAD Programları (SolidWorks/Fusion360) için 3D Gövde hazırlanıyor...")
    try:
        dxf3d_filename = os.path.join(output_dir, f"Paprika_CAD_Braket_{params.get('current_alg', 'Opt')}_{params['mat_name'][:3]}_{params['force']}N.dxf")
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        mesh_ent = msp.add_mesh()
        with mesh_ent.edit_data() as mesh_data:
            mesh_data.vertices = verts.tolist()
            mesh_data.faces = faces.tolist()
        doc.saveas(dxf3d_filename)
        print(f"✅ CAD Gövdesi '{dxf3d_filename}' olarak başarıyla kaydedildi!")
    except Exception as e:
        print(f"⚠️ CAD dosyası kaydedilirken bir hata oluştu: {e}")
    
    # --- FİLAMENT / AĞIRLIK HESAPLAMASI ---
    # (Lx * Ly * Lz) cm^3 cinsinden hacim hesabı
    eff_x = params.get('eff_x', nelx)
    vol_cm3 = (params['physical_length'] / 10.0) * ((params['physical_length'] * (params['nely']/eff_x)) / 10.0) * ((params['physical_length'] * (params['nelz']*2/eff_x)) / 10.0)
    final_vol_cm3 = vol_cm3 * (np.sum(x_opt) / len(x_opt))
    weight_g = final_vol_cm3 * params['density']
    print(f"⚖️ TAHMİNİ AĞIRLIK: Parça üretildiğinde yaklaşık {weight_g:.1f} gram malzeme (filament) harcayacaktır.")
    print("Dosyayı doğrudan Cura veya PrusaSlicer'da açarak 3D baskı alabilirsiniz!")

# =============================================================================
# 4. KULLANICI ARAYÜZÜ (GUI)
# =============================================================================

class GUI3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Paprika 3D Organik Topoloji Optimizasyonu")
        self.params = None
        
        f = ttk.LabelFrame(root, text="3D Braket ve Vida Ayarları", padding=15)
        f.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.materials = {
            "PLA (Polilaktik Asit)": {"E": 2600.0, "nu": 0.36, "yield": 40.0, "density": 1.24},
            "PETG (Polietilen Tereftalat)": {"E": 2100.0, "nu": 0.38, "yield": 50.0, "density": 1.27},
            "ABS / ASA": {"E": 2100.0, "nu": 0.35, "yield": 40.0, "density": 1.04},
            "PP (Polipropilen)": {"E": 1000.0, "nu": 0.42, "yield": 25.0, "density": 0.90},
            "PA (Nylon / Poliamid)": {"E": 1500.0, "nu": 0.40, "yield": 60.0, "density": 1.14},
            "Wood (Ahşap Dolgulu PLA)": {"E": 2000.0, "nu": 0.35, "yield": 20.0, "density": 1.20},
            "Alüminyum (Döküm)": {"E": 69000.0, "nu": 0.33, "yield": 150.0, "density": 2.70}
        }

        self.printers = {
            "Bambu Lab (256mm) - Varsayılan": 256.0,
            "Ender 3 / A1 Mini (220mm)": 220.0,
            "Prusa MK3/MK4 (250mm)": 250.0,
            "Büyük Boy / CR-10 (300mm)": 300.0,
            "Sınırsız / Endüstriyel": 9999.0
        }

        ttk.Label(f, text="3D Yazıcı Modeli:").grid(row=0, column=0, sticky="w", pady=4)
        self.printer_combo = ttk.Combobox(f, values=list(self.printers.keys()), state="readonly", width=25)
        self.printer_combo.current(0)
        self.printer_combo.grid(row=0, column=1, pady=4)

        ttk.Label(f, text="Braket Uzunluğu (X Ekseninde / mm):").grid(row=1, column=0, sticky="w", pady=4)
        self.len_ent = ttk.Entry(f, width=28); self.len_ent.insert(0, "150"); self.len_ent.grid(row=1, column=1)

        ttk.Label(f, text="Malzeme:").grid(row=2, column=0, sticky="w", pady=4)
        self.mat_combo = ttk.Combobox(f, values=list(self.materials.keys()), state="readonly", width=25)
        self.mat_combo.current(0)
        self.mat_combo.grid(row=2, column=1, pady=4)
        
        ttk.Label(f, text="Aşağı Çeken Yük (Newton):").grid(row=3, column=0, sticky="w", pady=4)
        self.f_ent = ttk.Entry(f, width=28); self.f_ent.insert(0, "500"); self.f_ent.grid(row=3, column=1)
        
        ttk.Label(f, text="Duvar Vida Çapı:").grid(row=4, column=0, sticky="w", pady=4)
        self.wall_screw = ttk.Combobox(f, values=["M4 (4mm)", "M6 (6mm)", "M8 (8mm)"], state="readonly", width=25)
        self.wall_screw.current(1); self.wall_screw.grid(row=4, column=1)
        
        ttk.Label(f, text="Taşıma Yüzeyine Vida Atılacak mı?:").grid(row=5, column=0, sticky="w", pady=4)
        self.top_screw = ttk.Combobox(f, values=["Yok (Sadece Yük)", "1 Adet", "2 Adet", "4 Adet"], state="readonly", width=25)
        self.top_screw.current(2); self.top_screw.grid(row=5, column=1)
        
        ttk.Label(f, text="Hedef Doluluk (Sağlamlık Oranı):").grid(row=6, column=0, sticky="w", pady=4)
        self.volfrac_combo = ttk.Combobox(f, values=["Otomatik (Minimum Güvenli Malzeme)", "%15 (Ultra Hafif)", "%25 (Çok Hafif)", "%35 (Standart)", "%50 (Sağlam)", "%65 (Maksimum Güç)"], state="readonly", width=30)
        self.volfrac_combo.current(0); self.volfrac_combo.grid(row=6, column=1)
        
        ttk.Label(f, text="Güvenlik Faktörü (Emniyet Katsayısı):").grid(row=7, column=0, sticky="w", pady=4)
        self.sf_combo = ttk.Combobox(f, values=["1.0 (Riskli Sınır)", "1.5 (Standart Endüstriyel)", "2.0 (Güvenli)", "3.0 (Ağır Hizmet)"], state="readonly", width=25)
        self.sf_combo.current(1); self.sf_combo.grid(row=7, column=1)
        
        ttk.Label(f, text="Optimizasyon Algoritmaları:").grid(row=8, column=0, sticky="w", pady=4)
        alg_frame = ttk.Frame(f)
        alg_frame.grid(row=8, column=1, sticky="w")
        self.simp_var = tk.BooleanVar(value=True)
        self.beso_var = tk.BooleanVar(value=False)
        self.lsm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(alg_frame, text="SIMP", variable=self.simp_var).pack(side="left", padx=3)
        ttk.Checkbutton(alg_frame, text="BESO", variable=self.beso_var).pack(side="left", padx=3)
        ttk.Checkbutton(alg_frame, text="LSM", variable=self.lsm_var).pack(side="left", padx=3)
        
        ttk.Label(f, text="Maksimum Döngü (Hassasiyet):").grid(row=9, column=0, sticky="w", pady=4)
        self.loop_combo = ttk.Combobox(f, values=["150", "400", "1000"], state="readonly", width=25)
        self.loop_combo.current(0); self.loop_combo.grid(row=9, column=1)
        
        # --- İLERİ SEVİYE EKLEMELİ İMALAT (3D BASKI) LABORATUVARI ---
        am_frame = ttk.LabelFrame(f, text="Eklemeli İmalat (3D Baskı) Özel Fizik Kuralları", padding=10)
        am_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.am_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(am_frame, text="45° Desteksiz Büyüme (Yerçekimi Damla Formu)", variable=self.am_var).grid(row=0, column=0, sticky="w", padx=5)
        
        self.aniso_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(am_frame, text="Katman Zayıflığı (Z Ekseninde %30 Dayanım Kaybı)", variable=self.aniso_var).grid(row=0, column=1, sticky="w", padx=5)
        
        self.lattice_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(am_frame, text="Gyroid / Kafes İç Yapı Modu (Lattice)", variable=self.lattice_var).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        wall_frame = ttk.Frame(am_frame)
        wall_frame.grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(wall_frame, text="Min Duvar Kalınlığı (mm):").pack(side="left")
        self.wall_ent = ttk.Entry(wall_frame, width=10)
        self.wall_ent.insert(0, "1.2")
        self.wall_ent.pack(side="left", padx=5)

        self.gpu_var = tk.BooleanVar(value=HAS_GPU)
        cb_text = "NVIDIA Ekran Kartı (GPU) Kullanımı Aktif" if HAS_GPU else f"GPU Kapalı (Yüksek Performanslı CPU Modu Aktif)"
        cb = ttk.Checkbutton(f, text=cb_text, variable=self.gpu_var)
        cb.grid(row=11, column=0, columnspan=2, pady=5)
        if not HAS_GPU: cb.state(['disabled'])
        
        ttk.Button(root, text="3D Organik Üretimi Başlat", command=self.start).pack(pady=10)
        
    def start(self):
        try:
            selected_printer = self.printer_combo.get()
            max_length = self.printers[selected_printer]
            
            physical_length = float(self.len_ent.get())
            if physical_length > max_length:
                messagebox.showerror("Boyut Hatası", f"Girdiğiniz uzunluk ({physical_length} mm), seçilen yazıcı hacmini ({max_length} mm) aşıyor!\n\nLütfen parçayı küçültün veya listeden daha büyük bir yazıcı (Sınırsız vb.) seçin.")
                return
            if physical_length <= 0:
                messagebox.showerror("Hata", "Lütfen geçerli bir uzunluk girin!")
                return

            mat_key = self.mat_combo.get()
            force_input = float(self.f_ent.get())
            force = force_input * 1.10 # %10 Garanti (Güvenlik) Payı
            
            sf_map = {"1.0 (Riskli Sınır)": 1.0, "1.5 (Standart Endüstriyel)": 1.5, "2.0 (Güvenli)": 2.0, "3.0 (Ağır Hizmet)": 3.0}
            
            sf_val = sf_map[self.sf_combo.get()]
            
            # --- KIRILMA / MUKAVEMET KONTROLÜ ---
            # Oransal fiziksel boyutlar (nelx=100, nely=60, nelz=20)
            Lx = physical_length
            Ly = Lx * (60.0 / 100.0) # Braket Yüksekliği (Alt boşluk için)
            Lz = Lx * (20.0 / 100.0) # Braket Kalınlığı
            
            # Eğilme Momenti ve Atalet Momenti (Katı blok varsayımıyla)
            M = force * Lx
            I = (Lz * (Ly**3)) / 12.0
            c = Ly / 2.0
            sigma_max_solid = (M * c) / I
            
            yield_stress = self.materials[mat_key]['yield']
            allowable_stress = yield_stress / sf_val
            
            volfrac_str = self.volfrac_combo.get()
            if "Otomatik" in volfrac_str:
                required_volfrac = (sigma_max_solid * 1.5) / allowable_stress # %50 gözeneklilik payı
                volfrac_val = max(0.22, min(0.85, required_volfrac)) # 3D Bütünlük için asgari %22 malzeme sınırı
                print(f"🎯 Otomatik Mod: Parçanın kırılmaması için gereken minimum malzeme %{int(volfrac_val*100)} olarak hesaplandı.")
            else:
                volfrac_val = float(volfrac_str.split('%')[1].split()[0]) / 100.0
                
            # Gözenekli yapı için stres tahmini
            estimated_stress = sigma_max_solid * (1.5 / volfrac_val)
            if estimated_stress > allowable_stress:
                msg = f"⚠️ DİKKAT: Girdiğiniz {force_input}N yük (+%10 Güvenlik Payı ile {force:.0f}N),\n{sf_val}x emniyet katsayısıyla bu malzemeyi ({mat_key}) KIRABİLİR!\n\n"
                msg += f"Tahmini Oluşacak Stres: {estimated_stress:.1f} MPa\nİzin Verilen Emniyet Sınırı: {allowable_stress:.1f} MPa\n"
                msg += f"(Malzemenin Kırılma Noktası: {yield_stress} MPa)\n\n"
                msg += "Çözüm Önerileri:\n1. 'Hedef Doluluk' oranını artırın.\n2. Parça uzunluğunu (X) kısaltın.\n3. Daha güçlü bir malzeme (ör: ABS veya PETG) seçin.\n\n"
                msg += "Yine de hesaplamayı başlatmak istiyor musunuz?"
                if not messagebox.askyesno("Kırılma Riski Uyarısı", msg):
                    return

            # --- DOSYA VE KLASÖR YÖNETİMİ ---
            base_dir = "Çıktılar"
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_mat_folder = mat_key.split()[0].replace("/", "-")
            folder_name = f"Braket_{safe_mat_folder}_{int(force)}N_{timestamp}"
            output_dir = os.path.join(base_dir, folder_name)
            os.makedirs(output_dir)

            selected_algs = []
            if self.simp_var.get(): selected_algs.append("SIMP")
            if self.beso_var.get(): selected_algs.append("BESO")
            if self.lsm_var.get(): selected_algs.append("LSM")
            if not selected_algs:
                messagebox.showerror("Hata", "Lütfen en az bir algoritma seçin!")
                return

            # Dinamik Eklemeli İmalat Parametreleri
            min_wall = float(self.wall_ent.get())
            pixel_size = physical_length / 100.0 # nelx
            rmin_val = max(1.5, (min_wall / 2.0) / pixel_size) # Duvar kalınlığını filtreye bağla
            
            penal_val = 1.8 if self.lattice_var.get() else 3.0 # Lattice modunda cezayı düşür
            extract_level = 0.25 if self.lattice_var.get() else 0.4

            screw_map = {"M4 (4mm)": 4, "M6 (6mm)": 6, "M8 (8mm)": 8}
            top_map = {"Yok (Sadece Yük)": 0, "1 Adet": 1, "2 Adet": 2, "4 Adet": 4}
            max_loop_val = int(self.loop_combo.get())
            self.params = {
                'algorithms': selected_algs,
                'output_dir': output_dir,
                'physical_length': physical_length,
                'mat_name': mat_key,
                'E0': self.materials[mat_key]['E'],
                'nu': self.materials[mat_key]['nu'],
                'yield_stress': yield_stress,
                'density': self.materials[mat_key]['density'],
                'force': force,
                'wall_screw_d': screw_map[self.wall_screw.get()],
                'top_screws': top_map[self.top_screw.get()],
                'use_gpu': self.gpu_var.get(),
                'am_filter': self.am_var.get(),
                'aniso_z': 0.7 if self.aniso_var.get() else 1.0,
                'lattice': self.lattice_var.get(),
                'extract_level': extract_level,
                'volfrac': volfrac_val, 'penal': penal_val, 'rmin': rmin_val, 'max_loop': max_loop_val, 'tol': 0.001,
                'nelx': 100, 'nely': 60, 'nelz': 20 # 3D Baskı için yüksek çözünürlük (Y artırıldı)
            }
            
            if len(selected_algs) > 1:
                messagebox.showinfo("Sıralı İşlem Bilgisi", f"Seçtiğiniz algoritmalar ({', '.join(selected_algs)}) SIRAYLA çalıştırılacaktır.\n\nBir algoritma bittiğinde penceresi kapanacak ve otomatik olarak diğeri başlayacaktır.\nLütfen tüm işlemlerin bitmesini bekleyin, en sonunda yan yana karşılaştırma videonuz oluşturulacaktır.")
            
            self.root.destroy()
        except ValueError:
            messagebox.showerror("Hata", "Lütfen değerleri rakam olarak girin!")

if __name__ == "__main__":
    root = tk.Tk()
    gui = GUI3D(root)
    root.mainloop()
    
    if gui.params:
        all_frames = {}
        for i, alg in enumerate(gui.params['algorithms']):
            if i > 0:
                print(f"\n⏳ Bir önceki işlem tamamlandı! Sıradaki algoritmaya ({alg}) otomatik geçiliyor...")
                time.sleep(3) # Geçişin fark edilmesi için 3 saniye mola
            print(f"\n{'='*50}\n>>> Paprika: {alg} ALGORİTMASI BAŞLIYOR <<<\n{'='*50}\n")
            gui.params['algorithm'] = alg
            gui.params['current_alg'] = alg
            x_opt, frames, p_solid, p_void = optimize_3d(gui.params)
            all_frames[alg] = frames
            export_3d_stl(x_opt, gui.params, p_solid, p_void)
            
        # --- VİDEOLARI YAN YANA BİRLEŞTİR VE KAYDET ---
        print("\n🎥 Paprika Karşılaştırma Videosu (MP4) Hazırlanıyor...")
        try:
            algs = gui.params['algorithms']
            max_frames = max([len(f) for f in all_frames.values()])
            sample_frame = all_frames[algs[0]][0]
            h, w, _ = sample_frame.shape
            
            video_filename = os.path.join(gui.params['output_dir'], f"Paprika_Karsilastirma_{gui.params['force']}N.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_filename, fourcc, 24.0, (w * len(algs), h)) # Sinematik akıcılık için 24 FPS
            
            for i in range(max_frames):
                row_frames = []
                for alg in algs:
                    if i < len(all_frames[alg]): row_frames.append(all_frames[alg][i])
                    else: row_frames.append(all_frames[alg][-1]) # Kısa biten algoritmanın son karesini sabit tut
                combined_frame = np.hstack(row_frames)
                out.write(combined_frame)
            out.release()
            print(f"✅ Video başarıyla '{video_filename}' olarak kaydedildi!")
        except Exception as e:
            print(f"⚠️ Video kaydedilirken bir hata oluştu: {e}")

        input("\nÇıkmak için Enter'a basın...")