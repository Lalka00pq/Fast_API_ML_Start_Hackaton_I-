import requests


health_responce = requests.get("http://127.0.0.1:8000/health")
print(health_responce.json())

check_classes_info = requests.get("http://127.0.0.1:8000/get_classes_info")
print(check_classes_info.json())

path_to_image = r"boat49.png"
images = {"image": open(path_to_image, "rb")}
response = requests.post("http://127.0.0.1:8000/inference", files=images)
print(response.json())
