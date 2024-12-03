from api import get_status, control, take_picture, set_simulation, restart_simulation
from simulation import Simulator
from Astar_joao_2 import AStar
from adaptivempc import adaptivempc
import sys
import time
from functools import wraps
import os
import math


SIMULATION = False
SIMULATION_SPEED = 20
RESTART_SIMULATION = True
simulator = Simulator(SIMULATION_SPEED)
simulator.picture_taken = False

def throttle(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        elapsed = time.time()-start 
        total_time = 0.5
        if elapsed < total_time:
            time.sleep(total_time - elapsed)
        return output
    return wrapper

@throttle
def main(astar=None):
    tol_time = SIMULATION_SPEED
    if not astar:
        status = get_status()
        print('Calculating trajectory')
        square_size = 10800/math.ceil(10800/(600-tol_time*status['vy']))
        centers,trajectory =  adaptivempc(
            status['width_x'],
            status['height_y'],
            status['vx'],
            status['vy'],
            square_size)
        coverage={}
        for center in centers:
            coverage[(center[0] ,center[1])]=0
        for item in os.listdir('MELVIN'):
            if not item[0].isalnum():
                continue
            idx = int(item[:item.index('_')])
            key = [*coverage][idx]
            coverage[key] = 1

        astar = AStar([0, 1], coverage, trajectory, square_size)
        astar.current_state = 7
        start_state = (
                astar.get_next_idx(
                        status['width_x'],
                        status['height_y'],
                        status['vx'],
                        status['vy']
                        ),
                tuple([*coverage.values()]),
                status['battery']/100,
                7,
                0)
        print('Searching')
        search(astar,start_state,tol_time)  
        return astar
    
    status = get_status()
    x = status["width_x"]
    y = status["height_y"]
    vx = status["vx"]
    vy = status["vy"]
    state = status['state']
    pos, action, start_time = astar.plan[0]
    y_dif = pos[1]-y
    y_dif = y_dif if y_dif>0 else y_dif+10800
    tol = vy*tol_time
    print(state)
    print('Battery:', status["battery"])
    print([*astar.coverage].index(astar.get_next_idx(pos[0],pos[1],vx,vy,-1)),"next action", astar.plan[0][:-1] ,"x:",x,"y:",y)
    
    print(f'T{round(time.time()-start_time,1)}')
    print(y_dif, tol)
    # if time.time()>start_time:
    #     del astar.plan[0]
    #     control(vx,vy,status["angle"], "charge")
    
    if  (state == 'safe' and astar.current_state != 5) or status["battery"] < 30:
        if status["battery"] < 10:
            control(vx,vy,status["angle"], "charge")
            time.sleep((3*60+90*5)/SIMULATION_SPEED)
            astar.current_state = 4
        else:
            astar.current_state = 5
        status = get_status()
        x = status["width_x"]
        y = status["height_y"]
        start_state = (
            astar.get_next_idx(x,y,vx,vy),
            tuple([*astar.coverage.values()]),
            status['battery']/100,
            astar.current_state,
            0)
        search(astar,start_state,tol_time) 
        return astar
    
    if action == 3:
        print('Changing to Charge\n\n')
        control(vx,vy,status["angle"], "charge")
        astar.current_state = 4
    elif y_dif<tol or (time.time()-start_time > -2 and time.time()-start_time < 0):
        if action==0:
            astar = picture(x,y,astar,pos, charge = [vx,vy,"narrow", "charge"])
            print('Changing to Charge\n\n')
            astar.current_state = 4
        elif action==1:
            print('Changing to Acquisition\n\n')
            control(vx,vy,"narrow", "acquisition")
            astar.current_state = 0
        elif action==2:
            astar = picture(x,y,astar,pos)
    elif time.time()<=start_time:
        return astar
    
    if time.time()>start_time:
        status = get_status()
        x = status["width_x"]
        y = status["height_y"]
        start_state = (
            astar.get_next_idx(x,y,vx,vy),
            tuple([*astar.coverage.values()]),
            status['battery']/100,
            astar.current_state,
            0)
    else:
        next_pos = astar.plan[1][0]
        idx = [*astar.coverage].index(astar.get_next_idx(next_pos[0],next_pos[1],vx,vy,-1))
        start_state = (
            [*astar.coverage][idx],
            tuple([*astar.coverage.values()]),
            status['battery']/100,
            astar.current_state,
            0)
    search(astar,start_state,tol_time) 
    
    
    return astar

def search(astar,start_state,tol_time):
    print('\n','CALCULATING PATH','\n')           
    path,_ = astar.search(start_state,tol_time)
    status = get_status()
    astar.create_plan(path,tol_time,[status['width_x'],status['height_y']],SIMULATION_SPEED)
    return astar 

def picture(x,y,alg,pos,format='jpeg', charge=None):
    print('\n')
    idx = [*alg.coverage].index(pos)
    if alg.coverage[pos] == 0 and take_picture(idx,x,y,format,charge):
        alg.coverage[pos] = 1
    elif charge and alg.coverage[pos] == 1:
        control(*charge)
    return alg
    

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
            print('Restarting')
            restart_simulation()
        set_simulation(SIMULATION,SIMULATION_SPEED)
        print("--Agent started--")
        astar = None
        while True:
            try:
                astar = main(astar)
            except TimeoutError:
                pass
    except KeyboardInterrupt:
        print('\n--Agent quit--')
        sys.exit(0)