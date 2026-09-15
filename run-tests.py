import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='unittests', pattern='*Tests.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)