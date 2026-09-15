import unittest
import os
from model import model_train, model_load, model_predict

class ModelTest(unittest.TestCase):
    def test_01_train(self):
        models = model_train(data_dir=os.path.join(".", "cs-train"), test=True)
        self.assertIn('all', models)
        
    def test_02_load(self):
        model = model_load('all', test=True)
        self.assertIsNotNone(model)
        
    def test_03_predict(self):
        res = model_predict('all', 2018, 12, 1, test=True)
        self.assertIn('y_pred', res)
        self.assertIsInstance(res['y_pred'], float)

if __name__ == '__main__':
    unittest.main()