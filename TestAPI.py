import requests

url = 'http://127.0.0.1:5000/predict' #This is the endpoint
data = {
    'text': 'Here you can insert your text',
    'model': 'logistic_regression' #Options here are logistic_regression or bert
}

response = requests.post(url, json=data) #Make the request
print(response.json()) #Prints the results
