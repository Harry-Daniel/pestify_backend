import os
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from tensorflow.lite.python.interpreter import Interpreter

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'tflite_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_float32.tflite')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.txt')

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    # fallback (keep order consistent with training!)
    return ['diamondback_moth','unknown','whitefly','aphid','armyworm','grasshopper']

CLASS_NAMES = load_labels()

# Load model once
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_info = interpreter.get_input_details()[0]
out_info = interpreter.get_output_details()[0]
IN_H, IN_W = int(in_info['shape'][1]), int(in_info['shape'][2])  # e.g., 224x224
IN_DTYPE = in_info['dtype']  # should be <class 'numpy.float32'> per your debug

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

@csrf_exempt
def predict_pest(request):
    if request.method != 'POST' or not request.FILES.get('image'):
        return JsonResponse({'error': 'POST request with image required'}, status=400)

    img_file = request.FILES['image']
    img_path = default_storage.save('uploads/' + img_file.name, img_file)
    full_path = os.path.join('media', img_path)

    try:
        # === PREPROCESS ===
        # Model wants raw 0–255 float32, RGB, plain resize
        img = Image.open(full_path).convert('RGB').resize((IN_W, IN_H), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32)        # keep 0–255 range
        # DO NOT divide by 255.0
        x = np.expand_dims(x, axis=0)               # shape (1, H, W, 3)

        # === INFERENCE ===
        interpreter.set_tensor(int(in_info['index']), x)
        interpreter.invoke()
        preds = interpreter.get_tensor(int(out_info['index']))[0]  # (C,)

        # === POSTPROCESS ===
        probs = softmax(preds)                       # model head behaves like softmax
        C = int(probs.shape[0])
        if len(CLASS_NAMES) != C:
            return JsonResponse({
                'error': f'Label count {len(CLASS_NAMES)} != model outputs {C}. '
                         f'Use labels.txt from the export.'
            }, status=500)

        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_label = str(CLASS_NAMES[top_idx])
        top_conf = float(probs[top_idx])

        top_k = min(3, C)
        top3 = [
            {'label': str(CLASS_NAMES[int(i)]), 'confidence': float(probs[int(i)])}
            for i in order[:top_k]
        ]

        return JsonResponse({
            'label': top_label,
            'confidence': top_conf,
            'top_k': top3
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    finally:
        try:
            os.remove(full_path)
        except Exception:
            pass
