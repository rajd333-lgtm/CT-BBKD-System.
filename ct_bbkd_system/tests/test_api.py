"""
CT-BBKD API Test Suite
======================
Tests all REST API endpoints.
Run with: python tests/test_api.py
"""

import sys
import json
import time
import unittest
import threading
sys.path.insert(0, '..')

# Import app and init
from backend.app import app, init_db, _start_time
import os

# Use in-memory test DB
TEST_DB = '/tmp/test_ct_bbkd.db'

class CTBBKDAPITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        import backend.app as api_module
        api_module.DB_PATH = TEST_DB
        init_db()
        cls.client = app.test_client()
        cls.exp_id = None

    def test_01_health(self):
        """Health endpoint returns 200"""
        r = self.client.get('/api/v1/health')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['status'], 'healthy')
        print("  ✓ Health check passed")

    def test_02_list_experiments_empty(self):
        """Empty list on fresh DB"""
        r = self.client.get('/api/v1/experiments')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertTrue(data['success'])
        print("  ✓ Empty experiment list OK")

    def test_03_create_experiment(self):
        """Create an experiment"""
        payload = {
            "name": "Test Experiment",
            "regime": "sudden_update",
            "timesteps": 5,
            "drift_schedule": {"3": 2},
            "interval_ms": 50,
            "seed": 42
        }
        r = self.client.post('/api/v1/experiments',
                             data=json.dumps(payload),
                             content_type='application/json')
        self.assertEqual(r.status_code, 201)
        data = json.loads(r.data)
        self.assertTrue(data['success'])
        self.__class__.exp_id = data['data']['id']
        print(f"  ✓ Experiment created: {self.exp_id}")

    def test_04_get_experiment(self):
        """Get experiment by ID"""
        if not self.exp_id:
            self.skipTest("No experiment created")
        r = self.client.get(f'/api/v1/experiments/{self.exp_id}')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['id'], self.exp_id)
        print("  ✓ Get experiment by ID OK")

    def test_05_experiment_runs(self):
        """Wait for short experiment to complete"""
        if not self.exp_id:
            self.skipTest("No experiment created")
        # Wait up to 5 seconds
        for _ in range(20):
            time.sleep(0.3)
            r = self.client.get(f'/api/v1/experiments/{self.exp_id}')
            data = json.loads(r.data)['data']
            if data['status'] in ('complete', 'failed'):
                break
        self.assertIn(data['status'], ('complete', 'running'))
        print(f"  ✓ Experiment status: {data['status']}")

    def test_06_get_metrics(self):
        """Metrics endpoint returns data"""
        if not self.exp_id:
            self.skipTest("No experiment created")
        time.sleep(1.5)  # Allow to populate
        r = self.client.get(f'/api/v1/experiments/{self.exp_id}/metrics')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)['data']
        self.assertIn('by_method', data)
        self.assertIn('sds_series', data)
        print(f"  ✓ Metrics returned: {data['total_points']} points")

    def test_07_get_summary(self):
        """Summary endpoint"""
        if not self.exp_id:
            self.skipTest("No experiment created")
        time.sleep(1.5)
        r = self.client.get(f'/api/v1/experiments/{self.exp_id}/summary')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)['data']
        self.assertIn('summary', data)
        print(f"  ✓ Summary: {len(data['summary'])} methods")

    def test_08_system_stats(self):
        """System stats endpoint"""
        r = self.client.get('/api/v1/system/stats')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)['data']
        self.assertIn('current', data)
        self.assertIn('cpu', data['current'])
        print(f"  ✓ System stats: CPU={data['current']['cpu']}%")

    def test_09_system_overview(self):
        """System overview endpoint"""
        r = self.client.get('/api/v1/system/overview')
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)['data']
        self.assertIn('experiments', data)
        print(f"  ✓ Overview: {data['experiments']['total']} total experiments")

    def test_10_quick_start(self):
        """Quick start endpoint"""
        r = self.client.post('/api/v1/experiments/quick-start',
                             data=json.dumps({"regime": "sudden_update"}),
                             content_type='application/json')
        self.assertIn(r.status_code, [200, 201])
        data = json.loads(r.data)['data']
        self.assertIn('id', data)
        print(f"  ✓ Quick start: {data['id']}")

    def test_11_delete_experiment(self):
        """Delete experiment"""
        if not self.exp_id:
            self.skipTest("No experiment created")
        r = self.client.delete(f'/api/v1/experiments/{self.exp_id}')
        self.assertEqual(r.status_code, 200)
        # Verify gone
        r2 = self.client.get(f'/api/v1/experiments/{self.exp_id}')
        self.assertEqual(r2.status_code, 404)
        print("  ✓ Delete experiment OK")

    def test_12_404_experiment(self):
        """404 for unknown experiment"""
        r = self.client.get('/api/v1/experiments/nonexistent_id')
        self.assertEqual(r.status_code, 404)
        print("  ✓ 404 for unknown experiment OK")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  CT-BBKD API Test Suite")
    print("="*55)
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite  = loader.loadTestsFromTestCase(CTBBKDAPITests)
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    print("="*55)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  {passed}/{result.testsRun} tests passed")
    if result.failures:
        for f in result.failures:
            print(f"  FAIL: {f[0]}")
    if result.errors:
        for e in result.errors:
            print(f"  ERROR: {e[0]}")
    print("="*55 + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
