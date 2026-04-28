import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import app as m
print('imported app')
print('loading', m.load_and_prepare_data())
print('preprocessing', m.preprocess_data())
print('trained', m.trained)
client = m.app.test_client()
payload = {
    'Category': 'Groceries',
    'Region': 'North',
    'Inventory Level': 200,
    'Price': 12.0,
    'Discount': 0,
    'Weather Condition': 'Sunny',
    'Holiday/Promotion': 0,
    'Competitor Pricing': 10.0,
    'Seasonality': 'Spring'
}
print('payload', payload)
resp = client.post('/api/predict', json=payload)
print('status', resp.status_code)
print('data', resp.get_data(as_text=True))
