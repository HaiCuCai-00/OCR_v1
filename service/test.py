import base64
import time

import requests

if __name__ == "__main__":
    URL = "http://127.0.0.1:8000/detection/"
    file = open("../data/images/cmnd_2/1.png", "rb")
    file1 = open("../data/images/cmnd_2/2.png", "rb")
    encoded_string = base64.b64encode(file.read())
    encoded_string1 = base64.b64encode(file1.read())
    PARAMS = {
        "abc123": {
            "type": "cmnd", 
            "images": [encoded_string, encoded_string1],
        }
    }
    for i in range(0, 10):
        start_time = time.time()
        res = requests.post(url=URL, json=PARAMS)
        print(res.json())
        end_time = time.time()
        print(f"Executed in {end_time - start_time} s")
