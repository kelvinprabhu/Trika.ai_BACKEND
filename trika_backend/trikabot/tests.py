# from django.test import TestCase

# Create your tests here.

import requests

url = "http://172.24.112.1:3000/api/user/complete-info"
url_custom = "http://172.24.112.1:3000/api/custom-challenges?userEmail=kelvinprabhu2071@gmail.com"
params = {"email": "kelvinprabhu2071@gmail.com"}

response = requests.get(url_custom)
print(response.json())
