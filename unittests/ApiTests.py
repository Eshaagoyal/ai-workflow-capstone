import unittest
import json
from app import app

class ApiTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_predict_endpoint(self):
        payload = {
            "mode": "test",
            "query": {"country": "all", "year": 2018, "month": 11, "day": 20}
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('y_pred', json.loads(response.data))

if __name__ == '__main__':
    unittest.main()