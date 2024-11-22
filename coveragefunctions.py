import numpy as np
import hashlib
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import shapely


def discretize_domain(y=10800,x=21600,pixels=600):
    """
        Divides a rectangular domain into sections of small squares
        x: integer that gives the width of the rectangular domain 
        y: integer that gives the height of the rectangular domain
        pixels: side of the discretization (width=height)

        Returns list of polygons each one corresponding to a domain section
    """

    radius = pixels/2

    num_squares_y = y//pixels
    num_squares_x = x//pixels

    grid = [] #stores all the polygons

    ##create division polygons
    for nx in range(num_squares_x):
        center_x = (nx+1)*radius
        for ny in range(num_squares_y):
            center_y = (ny+1)*radius
            rectangle_bounds = Polygon([(center_x+radius,center_y+radius),(center_x-radius,center_y+radius),
                (center_x-radius,center_y-radius),(center_x+radius,center_y-radius)])
            grid.append(rectangle_bounds)
    
    return grid


def covered_area(grid,alb,mode='narrow'):
    '''
    Returns dictionary where each section of the domain is associate with a percentual coverage
    '''
    if mode=='narrow':
        factor = 1
    elif mode =='normal':
        factor = 0.5
    elif mode == 'wide':
        factor = 0.33
    covered_area_by_sec = {}
    total_alb = shapely.unary_union(alb)
    for sect in grid:
        covered_area_by_sec[sect] = (intersection(sect,total_alb).area/sect.area)*factor*100

    return covered_area_by_sec

    
    
def coverage(sample,album ,mode='narrow'):
    """
        Returns a float that represents the incremental percentual area covered by taking a photo in a state.
        
        sample: Input state (x, y, m, e).
        mode: Coverage mode ('narrow', 'normal', 'wide').
        album: Vector that saves all photos as Polygons from shapely class
        

    """
    # Define radius based on mode
    if mode == 'narrow':
        radius = 300
    elif mode == 'normal':
        radius = 400
    elif mode == 'wide':
        radius = 500
    else:
        raise ValueError(f"Unknown mode: {mode}")

    area_to_subtract = 0
    x = sample[0]
    y = sample[1]
    #Square corner
    photo_boundaries = Polygon([(x+radius,y+radius),(x-radius,y+radius),
                (x-radius,y-radius),(x+radius,y-radius)])
        
    hyp_album = album
    hyp_album.append(photo_boundaries)
    current_album_area = total_area(album)


    return ((current_album_area+4*radius**2-total_area(hyp_album))*100)/(21600*10800)



def album_update(album,photo):
     '''Updates album with a photo
     photos: Polygon from shapely class
     album: List of Polygons from shapely class
     '''
     album.append(photo)
     return album

def total_area(album):
    '''This function returns the total area covered by polygons'''
    
    return shapely.unary_union(album).area

def to_photo(sample,mode='narrow'):
    '''Transforms a shot in coordinates (x,y) into a
     Polygon from shapely class (photo)
    '''

    if mode == 'narrow':
        radius = 300
    elif mode == 'normal':
        radius = 400
    elif mode == 'wide':
        radius = 500
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    x,y,_,_ = sample
    
    photo_boundaries = Polygon([(x+radius,y+radius),(x-radius,y+radius),
                (x-radius,y-radius),(x+radius,y-radius)])

    return photo_boundaries

#Initialize an empty album
# album = []

# sample_states = [(2, 3, 1.0, 1.0),(1,5,1.0,1.0)]
# for state in sample_states:
#     photo = to_photo(state,mode='narrow')


#     # album = album_update(album,photo)

#     area = total_area(album)

#     print(area,'This is the total area covered')

# sample_state_2 = (60,4,1.0,1.0)

# print('This would be the additional coverage by covering with the narrow mode the sample_state2', coverage(sample_state_2,album))
      
# print('This would be the additional coverage by covering with the wide mode the sample_state2', coverage(sample_state_2,album,mode='wide'))