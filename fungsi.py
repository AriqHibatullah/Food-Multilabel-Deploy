import cv2
import numpy as np
from rembg import remove
from skimage.feature import hog, local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

label_names = ['mie', 'telur', 'tomat', 'nasi', 'ikan', 'ayam', 'timun', 'selada']

def ekstrak_color(image, bins=(8,8,8), grid=(2,2)):
    h, w = image.shape[:2]
    grid_h, grid_w = grid
    step_h = h // grid_h
    step_w = w // grid_w

    hist_features = []

    for row in range(grid_h):
        for col in range(grid_w):
            patch = image[row*step_h:(row+1)*step_h, col*step_w:(col+1)*step_w]

            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_patch, 1, 255, cv2.THRESH_BINARY)

            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_patch], [0,1,2], mask, bins, [0,180,0,256,0,256])
            cv2.normalize(hist, hist)
            hist_features.extend(hist.flatten())

    return np.array(hist_features)

def ekstrak_lbp(image, points_list=[8,16,24], radius_list=[1,2,3], method='uniform'):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    all_hist = []
    for points, radius in zip(points_list, radius_list):
        lbp = local_binary_pattern(gray, points, radius, method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        all_hist.extend(hist)

    return np.array(all_hist)

def ekstrak_glcm(image, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.max() > levels - 1:
        gray = (gray / (gray.max() / (levels - 1))).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)

    glcm = graycomatrix(gray,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    fitur_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    fitur = []
    for prop in fitur_names:
        nilai = graycoprops(glcm, prop)
        fitur.extend(nilai.flatten())

    return np.array(fitur)

def remove_background(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = remove(img_rgb)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    return result_bgr

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True
}

def ekstrak_fitur(img, pca, scaler_hist, scaler_hog, scaler_lbp, scaler_glcm):
    hist = ekstrak_color(img)
    hist_scaled = scaler_hist.transform([hist])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, **hog_params)
    hog_scaled = scaler_hog.transform([hog_feat])
    hog_pca = pca.transform(hog_scaled)

    lbp_feat = ekstrak_lbp(img)
    lbp_scaled = scaler_lbp.transform([lbp_feat])

    glcm_feat = ekstrak_glcm(gray)
    glcm_scaled = scaler_glcm.transform([glcm_feat])

    fitur = np.hstack([hist_scaled[0], hog_pca[0], lbp_scaled[0], glcm_scaled[0]])

    return fitur

def ekstrak_resnet50(image):
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    fitur = base_model.predict(img_array)

    return fitur

def modeling(fitur) :
    probs = model.predict_proba(fitur)[0]
    pred = (probs >= thresholds).astype(int)
    pred_labels = [label for label, val in zip(label_names, pred) if val == 1]

    df_terpilih = df[df['nama'].isin(pred_labels)]
    gizi_total = df_terpilih.drop(columns='nama').sum(numeric_only=True)

    return gizi_total, pred_labels

def prediksi_gizi(image, model, thresholds, df, pca, scaler_hist, scaler_hog, scaler_lbp, scaler_glcm):
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (224, 224))
    image = remove_background(image)

    fitur = None
    if model == 'model':
        fitur = ekstrak_fitur(image, pca, scaler_hist, scaler_hog, scaler_lbp, scaler_glcm).reshape(1, -1)
    elif model == 'model_pro':
        fitur = ekstrak_resnet50(image)
        
    gizi_total, pred_labels = modeling(fitur)
    
    return gizi_total, pred_labels, image

def hitung_akg(gender, berat, tinggi, usia, aktivitas_input) :
    if gender == 'Pria':
        bmr = 66 + (13.7 * berat) + (5 * tinggi) - (6.8 * usia)
    elif gender == 'Wanita':
        bmr = 655 + (9.6 * berat) + (1.8 * tinggi) - (4.7 * usia)
    else:
        return None, None, None, None

    aktivitas_map = {'Tidak aktif': 1.2, 'Sedikit aktif': 1.375, 'Cukup aktif': 1.55, 'Aktif': 1.725, 'Sangat aktif': 1.9}
    aktivitas = aktivitas_map.get(aktivitas_input, 1.2)
    total_kalori = round(bmr * aktivitas)

    karbo_kal = total_kalori * 0.60
    protein_kal = total_kalori * 0.15
    lemak_kal = total_kalori * 0.25

    kebutuhan_karbo = round(karbo_kal / 4, 2)
    kebutuhan_protein = round(protein_kal / 4, 2)
    kebutuhan_lemak = round(lemak_kal / 9, 2)

    return total_kalori, kebutuhan_karbo, kebutuhan_protein, kebutuhan_lemak

def get_gap(progress, target):
    return {k: target[k] - progress[k] for k in target}

def pilih_makanan(label, df, progress, target, taken, n_top=10):
    kandidat = df[(df['label'] == label) & (~df['Nama'].isin(taken))].copy()
    if kandidat.empty:
        return None

    gap = get_gap(progress, target)
    zat_prioritas = max(gap, key=lambda k: abs(gap[k]))
    kandidat['selisih'] = kandidat[zat_prioritas].apply(lambda x: abs(x - gap[zat_prioritas]))

    kandidat_sorted = kandidat.sort_values('selisih').head(n_top)
    return kandidat_sorted.sample(1).iloc[0]

def sudah_cukup(progress, toleransi):
    return all(toleransi[k][0] <= progress[k] <= toleransi[k][1] for k in toleransi)

def proporsi(progress, target):
    return {k: progress[k] / target[k] for k in progress}

def rekomendasi_makanan(df, target_k, target_p, target_l, aktual_k, aktual_p, aktual_l) :
    target = {'karbohidrat': target_k, 'protein': target_p, 'lemak': target_l}
    toleransi = {key: (0.8 * val, val) for key, val in target.items()}
    progress = {'karbohidrat': aktual_k, 'protein': aktual_p, 'lemak': aktual_l}
    kombinasi = []
    taken = []
    counter_label = {'makanan energi': 0, 'makanan pembentuk otot': 0}
    while not sudah_cukup(progress, toleransi):
        gap = get_gap(progress, target)
        zat_dominan = max(gap, key=lambda k: abs(gap[k]))

        if zat_dominan == 'karbohidrat':
            label = 'makanan energi'
        elif zat_dominan == 'protein':
            label = 'makanan pembentuk otot'
        else:
            label = 'makanan ringan'

        if progress['lemak'] >= 0.7 * target['lemak']:
            label = 'makanan low-fat'

        if counter_label['makanan energi'] >= 2 :
            if counter_label['makanan pembentuk otot'] >= 2 :
                if progress['lemak'] >= 0.7 * target['lemak'] :
                    label = 'makanan low-fat'
                else :
                    label = 'makanan ringan'
            else :
                label = 'makanan pembentuk otot'

        makanan = pilih_makanan(label, df, progress, target, taken)
        if makanan is None:
            break

        simulasi_progress = progress.copy()
        for zat in simulasi_progress:
            simulasi_progress[zat] += makanan[zat]

        if any(simulasi_progress[k] > toleransi[k][1] for k in simulasi_progress):
            break

        kombinasi.append(makanan)
        taken.append(makanan['Nama'])

        for zat in progress:
            progress[zat] += makanan[zat]
        if label in counter_label:
            counter_label[label] += 1

    return kombinasi






