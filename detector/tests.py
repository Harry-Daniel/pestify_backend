import io

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image

from .views import CLASS_NAMES


def make_image(color=(34, 139, 34), size=(320, 240), fmt='JPEG'):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format=fmt)
    return SimpleUploadedFile('leaf.jpg', buf.getvalue(), content_type='image/jpeg')


class PredictPestTests(TestCase):
    url = '/api/predict/'

    def test_get_is_rejected(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 400)

    def test_post_without_image_is_rejected(self):
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 400)

    def test_non_image_upload_is_rejected(self):
        bogus = SimpleUploadedFile('notes.txt', b'not an image', content_type='text/plain')
        response = self.client.post(self.url, {'image': bogus})
        self.assertEqual(response.status_code, 400)

    def test_prediction_response_shape(self):
        response = self.client.post(self.url, {'image': make_image()})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn(data['label'], CLASS_NAMES)
        self.assertGreaterEqual(data['confidence'], 0.0)
        self.assertLessEqual(data['confidence'], 1.0)
        self.assertEqual(len(data['top_k']), 3)
        self.assertEqual(data['top_k'][0]['label'], data['label'])
        confidences = [p['confidence'] for p in data['top_k']]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_png_with_alpha_is_accepted(self):
        buf = io.BytesIO()
        Image.new('RGBA', (100, 100), (200, 50, 50, 128)).save(buf, format='PNG')
        upload = SimpleUploadedFile('leaf.png', buf.getvalue(), content_type='image/png')
        response = self.client.post(self.url, {'image': upload})
        self.assertEqual(response.status_code, 200)
