from api import get_status, control, take_picture, set_simulation, restart_simulation
import sys
import time

SIMULATION = False
SIMULATION_SPEED = 20
RESTART_SIMULATION = True

def main():
    time.sleep(1/SIMULATION_SPEED)
    status = get_status()
    print(status["state"])
    print("x:",status["width_x"],"y:",status["height_y"])
    if status["state"] == "safe":
        control(status["vx"],status["vy"],status["angle"], "charge")
    if status["state"] == "deployment":
        control(status["vx"],status["vy"],status["angle"], "acquisition")
    if status["state"] == "acquisition":
        take_picture(status["width_x"],status["height_y"])
        control(status["vx"],status["vy"],status["angle"], "charge")
        

# Example status:
# {
#     "state": "deployment",
#     "angle": "normal",
#     "simulation_speed": 1,
#     "width_x": 7653,
#     "height_y": 5108,
#     "vx": 4.35,
#     "vy": 5.49,
#     "battery": 100.0,
#     "max_battery": 100.0,
#     "fuel": 100.0,
#     "distance_covered": 24.5,
#     "area_covered": {"narrow": 0.0,
#     "normal": 0.0,
#     "wide": 0.0},
#     "data_volume": {"data_volume_sent": 100,
#     "data_volume_received": 131},
#     "images_taken": 0,
#     "active_time": 0.0,
#     "objectives_done": 0,
#     "objectives_points": 0,
#     "timestamp": "2024-11-16T09:33:15.827616Z"
# }

if __name__ == '__main__':
    try:
        if RESTART_SIMULATION:
            restart_simulation()
        set_simulation(SIMULATION,SIMULATION_SPEED)
        print("Agent started.")
        while True:
            main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)