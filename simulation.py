from api import get_status
from datetime import datetime, timezone, timedelta
import numpy as np
import shutil
import os

class Simulator:
    def __init__(self, simulation_speed) -> None:
        self.simulation_speed = simulation_speed
        self.slots = None
        
        self._setup_communication()
        self.get_slots()
    
    def get_slots(self):
        current_utc_time = datetime.now(timezone.utc)
        time_difference = current_utc_time - self.first_pass_start
        
        block_minutes = 90/self.simulation_speed
        pass_minutes = 10/self.simulation_speed
        block_duration = timedelta(minutes=block_minutes)
        pass_duration = timedelta(minutes=pass_minutes)
        
        num_blocks, remainder = divmod(time_difference, block_duration)
        if  remainder.total_seconds() / 60 < pass_minutes:
            start = current_utc_time - remainder
        else:
            start = current_utc_time + block_duration - remainder

        if not self.slots or num_blocks > self.slots['slots'][-1]['id']:
            output = {'slots':[]}
            for i in range(30):
                output['slots'].append({
                    'id': num_blocks + i,
                    'start': self._dt2str(start),
                    'end': self._dt2str(start + pass_duration),
                    'enabled': False
                })
                start = start + block_duration
            
            self.slots = output
        else:
            old_start = self.slots['slots'][0]['id']
            
            if num_blocks > old_start:
                if self._str2dt(self.slots['slots'][num_blocks-old_start-1]['end']) > current_utc_time:
                    num_blocks += -1
                self.slots['slots'] = self.slots['slots'][num_blocks-old_start:]
                start = start + timedelta(minutes=block_minutes*(30-num_blocks+old_start))
                for i in range(old_start,num_blocks):
                    self.slots['slots'].append({
                        'id': i+30,
                        'start': self._dt2str(start),
                        'end': self._dt2str(start + pass_duration),
                        'enabled': False
                    })
                    start = start + block_duration
        
        return self.slots
    
    def book_slot(self, id, enabled):
        error_message = 'Trying to book invalid id'
        self.get_slots()
        first_id = self.slots['slots'][0]['id']
        if any([id < first_id,
                id > first_id + 30,
                self.slots['slots'][id-first_id]['id'] != id]):
            print(error_message)
            return None
        
        self.slots['slots'][id-first_id]['enabled'] = enabled
        
    def transfer_images(self) -> bool:
        first_slot = self.get_slots()['slots'][0]
        start = self._str2dt(first_slot['start'])
        current_utc_time = datetime.now(timezone.utc)
        
        if get_status()['state'] != 'communication':
            print('Not in communication mode!')
            return False
        if not all([
                first_slot['enabled'],
                (current_utc_time-start).total_seconds() > 0
                ]):
            print('Not in communication slot!')
            return False
        
        if np.random.random() < 0.05:
            self.book_slot(self.get_slots()['slots'][0]['id'],False)
            print('Communication random fail!')
            return False
        
        for item in os.listdir('MELVIN'):
            source_path = os.path.join('MELVIN', item)
            destination_path = os.path.join('images', item)
            
            shutil.move(source_path, destination_path)
        print('Images downloaded successfully.')
        return True
            
    def _setup_communication(self)-> None:
        current_utc_time = datetime.now(timezone.utc)
        rand_start = np.random.randint(-80/self.simulation_speed,0)
        self.first_pass_start = current_utc_time + timedelta(minutes=rand_start)
    
    def _dt2str(self, dt):
        return str(dt.isoformat(timespec='microseconds')[:-6])+'Z'
    
    def _str2dt(self,string):
        dt = datetime.strptime(string, "%Y-%m-%dT%H:%M:%S.%fZ")
        return dt.replace(tzinfo=timezone.utc)
    
if __name__ == '__main__':
    # print(simulator.get_slots())
    # simulator.book_slot(3)
    pass