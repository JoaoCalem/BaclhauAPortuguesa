import requests
import json
from concurrent.futures import ThreadPoolExecutor

URI = 'http://10.100.50.1:33000'

def get_status():
    response = requests.get(f'{URI}/observation')
    if response.status_code == 200:
        return response.json()
    print_error(response, "Get status")
    return None

def control(vx, vy, angle, state):
    response = requests.put(f'{URI}/control',data = json.dumps({
            "vel_x": vx,
            "vel_y": vy,
            "camera_angle": angle,
            "state": state
        }))
    if response.status_code != 200:       
        print_error(response, "Control")

def take_picture(idx, format='tiff', charge=None) -> bool:
    def fetch_status():
        return get_status()

    def fetch_image():
        return requests.get(f'{URI}/image')

    with ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        status = executor.submit(fetch_status)
        future_image = executor.submit(fetch_image)
        status = status.result()
        # Retrieve results
        x = status['width_x']
        y = status['height_y']
        response = future_image.result()

    # Handle image saving
    if charge:
        control(*charge)
    if response.status_code == 200:
        with open(f"MELVIN/{idx}_{x}_{y}.{format}", "wb") as file:
            file.write(response.content)
        print(f"Image saved as {idx}_{x}_{y}.{format}")
        return True

    print_error(response, "Take picture")
    return False

def set_simulation(simulation, simulation_speed):
    response = requests.put(f'{URI}/simulation',params = {
            "is_network_simulation": simulation,
            "user_speed_multiplier": simulation_speed,
        })
    if response.status_code != 200:       
        print_error(response, "Set simulation")
        
        
def restart_simulation():
    response = requests.get(f'{URI}/reset')
    if response.status_code != 200:       
        print_error(response, "Restart simulation")

def print_error(response, name):
    print(f"{name} failed with the following error:")
    print(response.status_code)
    print(response.reason)