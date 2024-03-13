import unittest
from streamlit_app import make_prediction

class TestHelloApp(unittest.TestCase):
    
    def test_make_prediction(self):
        data = {
            'appt_date': '2021-01-01',
            'book_date': '2021-01-01',
            'lead_time': 10,
            'total_no_show': 2,
            'total_success': 5,
            'total_cancel': 1,
            'total_reschedule': 1
        }
        # Assume that the expected result with this data is 0
        self.assertEqual(make_prediction(data), 0)

        # Add more tests for other scenarios and edge cases

if __name__ == '__main__':
    unittest.main()
