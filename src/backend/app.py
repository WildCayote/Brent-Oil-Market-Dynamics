from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from utils.price_analyzing import (
    load_price_data,
    calculate_price_trends, 
    calculate_yearly_average_price, 
    calculate_analysis_metrics,calculate_price_distribution, 
    calculate_event_impact, get_prices_around_event
)

app = Flask(__name__)
CORS(app)

# Load the price data
price_data = load_price_data()

key_events = {
    "Russian Financial Crisis": "1999-08-17",
    "Hurricane Katrina": "2005-08-29",
    "Arab Spring": "2010-12-14",
    "U.S. shale oil boom": "2014-06-30",
    "Post-COVID-19 Recovery + OPEC+ Cuts": "2021-09-22",
}

@app.route('/price-trends', methods=['GET'])
def get_price_trends():
    trends_data = []
    for event, date in key_events.items():
        event_date = pd.to_datetime(date)
        prices_around_event = get_prices_around_event(event_date, price_data, days_before=180, days_after=180)
        trends_data.append({
            'event': event,
            'date': date,
            'prices': prices_around_event['Price'].tolist(),
            'dates': prices_around_event.index.tolist()
        })
    return jsonify(trends_data)

@app.route('/event-impact', methods=['GET'])
def get_event_impact():
    results = []
    for event, date in key_events.items():
        impact_data = calculate_event_impact(event, date, price_data)
        results.append(impact_data)
    return jsonify(results)

@app.route('/analysis-metrics', methods=['GET'])
def get_analysis():
    try:
        analysis_results = calculate_analysis_metrics(price_data.reset_index())
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

@app.route('/prices', methods=['GET'])
def get_price_trend():
    try:
        # Generate full dataset without filtering
        price_data_dict = calculate_price_trends(price_data)
        
        return jsonify(price_data_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/average-yearly-price', methods=['GET'])
def get_yearly_average():
    try:
        analysis_results = calculate_yearly_average_price(price_data)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
    
@app.route('/price-distribution', methods=['GET'])
def get_distribution():
    try:
        analysis_results = calculate_price_distribution(price_data)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500     
if __name__ == '__main__':
    app.run(debug=True)