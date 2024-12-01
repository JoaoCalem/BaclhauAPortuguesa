import requests
import json

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

def take_picture(idx, x, y, format='tiff', charge=None) -> bool:
    response = requests.get(f'{URI}/image')
    if response.status_code == 200:
        if charge:
            control(*charge)
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