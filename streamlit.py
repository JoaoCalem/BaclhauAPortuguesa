import streamlit as st
import requests
import json

st.write("Bacalhau A Portuguesa")
url = 'http://10.100.50.1:33000/'
    
with st.expander("Status"):
    if st.button('Update Status'):
        response = requests.get(f'{url}observation').json()
        with open('status.json', 'w') as f:
            json.dump(response, f)
    with open('status.json', 'r') as f:
        status = json.load(f)
    st.write(status)
    
with st.expander('Control'):
    states = ['acquisition', 'charge', 'communication']
    if status['state'] in states:
        ind = states.index(status['state'])
    else:
        ind = 0
    state = st.selectbox('State', states, ind)
    st.write(f'Current state: {status["state"]}')
    
    st.divider()
    
    angles = ['narrow', 'normal', 'wide']
    if status['angle'] in angles:
        ind = angles.index(status['angle'])
    else:
        ind = 0
    angle = st.selectbox('Angle', angles, ind)
    st.write(f'Current angle: {status["angle"]}')
    
    st.divider()

    vx = st.text_input('Vx', status["vx"])
    st.write(f'Current vx: {status["vx"]}')
    
    st.divider()

    vy = st.text_input('Vy', status["vy"])
    st.write(f'Current vy: {status["vy"]}')
    
    if st.button('Submit Control Request'):
        st.write(requests.put(f'{url}control',data = json.dumps({
            "vel_x": vx,
            "vel_y": vy,
            "camera_angle": angle,
            "state": state
        })).reason)

