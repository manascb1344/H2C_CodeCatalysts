from flask import Flask, request, jsonify
import pandas as pd

df = pd.read_csv("./HI-Small_Trans.csv")

app = Flask(__name__)

@app.route('/transactions', methods=['GET', 'POST'])
def get_transactions():
    if request.method == 'POST':
        data = request.json
        print("Received POST request data:", data)  
        
        if data is None or 'account_number' not in data:
            print("Error: Please provide account_number in the request body")  
            return jsonify({'error': 'Please provide account_number in the request body'}), 400
        
        account_number = data['account_number']
        print("Received account number:", account_number)  
    else:
        account_number = request.args.get('account_number')
        print("Received GET request account number:", account_number)
        if account_number is None:
            print("Error: Please provide account_number parameter")  
            return jsonify({'error': 'Please provide account_number parameter'}), 400
    
    transactions_from_account = df[df['Account'] == account_number].to_dict('records')
    transactions_to_account = df[df['Account.1'] == account_number].to_dict('records')
    
    print("Sending response...")  
    return jsonify({'transactions_from_account': transactions_from_account, 'transactions_to_account': transactions_to_account})

if __name__ == '__main__':
    app.run(debug=True)
