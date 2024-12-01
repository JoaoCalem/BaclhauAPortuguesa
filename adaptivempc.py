# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:42:02 2024

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import math
def calculate_trajectory(domain_width, domain_height, a_c, v_x0, v_y0, num_steps, dt):
    """
    Calculate the trajectory of a particle using the leapfrog method with a larger spatial step.

    Args:
    - domain_width: Width of the domain (pixels).
    - domain_height: Height of the domain (pixels).
    - a_c: Constant acceleration (pixels/second^2).
    - v_x0: Initial velocity in the x-direction (pixels/second).
    - v_y0: Initial velocity in the y-direction (pixels/second).
    - num_steps: Number of steps to simulate.
    - dt: Time step size.

    Returns:
    - trajectory: Array of (x, y) positions over time.
    """
    # Initialize position and velocity
    x, y = domain_width // 2, domain_height // 2  # Start at the center
    v_x, v_y = v_x0, v_y0
    trajectory = [(x, y)]

    # Compute initial half-step velocities
    a_x = np.sqrt(a_c**2 / (1 + (v_y / v_x)**2)) * np.sign(v_x)
    a_y = np.sqrt(a_c**2 / (1 + (v_x / v_y)**2)) * np.sign(v_y)
    v_x_half = v_x - 0.5 * a_x * dt
    v_y_half = v_y - 0.5 * a_y * dt

    for _ in range(num_steps):
        # Update position using half-step velocity
        x = x + v_x_half * dt
        y = y + v_y_half * dt

        # Teleport if out of bounds
        if x < 0:
            x = domain_width + x
        elif x >= domain_width:
            x = x - domain_width

        if y < 0:
            y = domain_height + y
        elif y >= domain_height:
            y = y - domain_height

        # Append position to trajectory
        trajectory.append((int(x), int(y)))

        # Compute acceleration
        a_x = np.sqrt(a_c**2 / (1 + (v_y_half / v_x_half)**2)) * np.sign(v_x_half)
        a_y = np.sqrt(a_c**2 / (1 + (v_x_half / v_y_half)**2)) * np.sign(v_y_half)

        # Update velocity
        v_x_half = v_x_half + a_x * dt
        v_y_half = v_y_half + a_y * dt

    return np.array(trajectory)



def plot_trajectory(domain_width, domain_height, trajectory):
    """
    Visualize the trajectory within the rectangular domain.

    Args:
    - domain_width: Width of the domain.
    - domain_height: Height of the domain.
    - trajectory: Array of (x, y) positions.
    """
    plt.figure(figsize=(12, 6))
    plt.xlim(0, domain_width)
    plt.ylim(0, domain_height)
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot the trajectory
    # plt.plot(trajectory[:, 0], trajectory[:, 1], '-r', label='Trajectory')
    plt.scatter(trajectory[:, 0], trajectory[:, 1], s=2, color='blue')

    plt.title("Trajectory (Leapfrog Method)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_border_intersections(domain_width, domain_height, x0, y0, v_x, v_y, proximity_threshold,square_size):
    """
    Calculate all intersection points of a trajectory with domain borders, including teleportation points.
    Stops when a new intersection is within the proximity threshold of a previous one.

    Args:
    - domain_width: Width of the domain (pixels).
    - domain_height: Height of the domain (pixels).
    - x0, y0: Initial position (pixels).
    - v_x, v_y: Velocity components (pixels/step).
    - proximity_threshold: Proximity threshold for stopping (pixels).

    Returns:
    - intersections: List of (x, y) intersection points.
    """
    intersections = []
    x, y = x0, y0

    while True:
        # Calculate alphas for hitting vertical and horizontal borders
        if v_x > 0:
            alpha_x = (domain_width - x) / v_x  # Hit the right border
        elif v_x < 0:
            alpha_x = -x / v_x  # Hit the left border
        else:
            alpha_x = float('inf')  # No movement in x

        if v_y > 0:
            alpha_y = (domain_height - y) / v_y  # Hit the top border
        elif v_y < 0:
            alpha_y = -y / v_y  # Hit the bottom border
        else:
            alpha_y = float('inf')  # No movement in y

        # Determine which border is hit first
        alpha = min(alpha_x, alpha_y)

        # Calculate the intersection point
        x_next = x + alpha * v_x
        y_next = y + alpha * v_y
        # Skip adding consecutive duplicate intersections
        # if len(intersections) == 0 or not np.allclose(intersections[-1], (x_next, y_next)):
        intersections.append((x_next, y_next))

        # Teleport to the opposite side and record 0-axis intersection
        if alpha == alpha_x:  # Vertical border
            if x_next >= domain_width:
                new_intersection = (0, y_next)  # Crossing to the left side
                if not np.allclose(intersections[-1], new_intersection):
                    intersections.append(new_intersection)
                x_next -= domain_width  # Teleport right to left
            elif x_next < 0:
                new_intersection = (domain_width, y_next)  # Crossing to the right side
                if not np.allclose(intersections[-1], new_intersection):
                    intersections.append(new_intersection)
                x_next += domain_width  # Teleport left to right

        if alpha == alpha_y:  # Horizontal border
            if y_next >= domain_height:
                new_intersection = (x_next, 0)  # Crossing to the bottom side
                if not np.allclose(intersections[-1], new_intersection):
                    intersections.append(new_intersection)
                y_next -= domain_height  # Teleport top to bottom
            elif y_next < 0:
                new_intersection = (x_next, domain_height)  # Crossing to the top side
                if not np.allclose(intersections[-1], new_intersection):
                    intersections.append(new_intersection)
                y_next += domain_height  # Teleport bottom to top

        # Check proximity to previous intersections
        for px, py in intersections[:-1]:
            distance = np.sqrt((x_next - px) ** 2 + (y_next - py) ** 2)
            if distance < proximity_threshold:
                intersections.append((x_next, y_next))
                return np.array(intersections)

        # Update position
        x, y = x_next, y_next

def find_distance_between_trajectories(intersections, domain_width=21600, domain_height=10800):
    """
    Find the maximum distance between consecutive points on the same axis in the trajectory.

    Args:
    - intersections: List of (x, y) intersection points.
    - domain_width: Width of the domain.
    - domain_height: Height of the domain.

    Returns:
    - max_distance: Maximum Euclidean distance between consecutive points on the same axis.
    """
    # Group intersections by axis
    left_border = [p[1] for p in intersections if p[0] == 0]
    right_border = [p[1] for p in intersections if p[0] == domain_width]
    bottom_border = [p[0] for p in intersections if p[1] == 0]
    top_border = [p[0] for p in intersections if p[1] == domain_height]

    # Sort each group
    left_border.sort(key=lambda p: p)  # Sort by y
    right_border.sort(key=lambda p: p)  # Sort by y
    bottom_border.sort(key=lambda p: p)  # Sort by x
    top_border.sort(key=lambda p: p)  # Sort by x

    # Calculate maximum distances
    avg_distance= 0
    count=0
    for border in [left_border, right_border, bottom_border, top_border]:
        for i in range(1, len(border)):
            a = border[i - 1]
            b= border[i]
            distance = abs(b-a)
            avg_distance +=  distance
            count+=1
    return avg_distance/count
def calculate_triangle_sides(avg_distance, trajectory_angle_radians):
    """
    Calculate the sides of a right triangle given the hypotenuse and angle.

    Args:
    - avg_distance: Hypotenuse of the triangle (average distance).
    - trajectory_angle_degrees: Angle of the trajectory in degrees.

    Returns:
    - adjacent: Length of the adjacent side.
    - opposite: Length of the opposite side.
    """
    # Calculate the sides of the triangle
    adjacent = avg_distance * math.cos(trajectory_angle_radians)
    opposite = avg_distance * math.sin(trajectory_angle_radians)

    return adjacent, opposite

def find_midpoint_of_trajectory(intersections):
    """
    Find the geometric midpoint of the trajectory using the intersection points.

    Args:
    - intersections: List of (x, y) intersection points.

    Returns:
    - midpoint: (x, y) coordinates of the geometric midpoint.
    """
    x1, y1 = intersections[0]  # Start of trajectory
    x2, y2 = intersections[-1]  # End of trajectory

    # Midpoint as the geometric center between the first and last points
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    return midpoint

def map_to_closest_value(value, options):
    """
    Map a given value to the closest in a list of options.

    Args:
    - value: Value to map.
    - options: List of possible values.

    Returns:
    - closest: Closest value from the options.
    """
    if value<=max(options):
        options = [x for x in options if x>=value]
    
    closest = min(options, key=lambda x: abs(value - x) )
    
    return closest
def place_squares_trajectory(trajectory_segments, square_size):
    """
    Place squares along the trajectory to ensure no overlap on the same line.

    Args:
    - trajectory_segments: List of line segments [(x1, y1, x2, y2)].
    - square_size: Size of the square.

    Returns:
    - square_centers: List of square centers [(x, y)].
    """
    square_centers = []
    half_size = square_size / 2

    for segment in trajectory_segments:
        x1, y1, x2, y2 = segment
        # Calculate the length of the segment
        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Calculate the direction vector (normalized)
        direction_vector = ((x2 - x1) / segment_length, (y2 - y1) / segment_length)

        # Number of squares needed
        num_squares = int(np.ceil(segment_length / square_size))

        # Place square centers along the segment
        for i in range(num_squares):
            center_x = x1 + (i * square_size + half_size) * direction_vector[0]
            center_y = y1 + (i * square_size + half_size) * direction_vector[1]
            square_centers.append((center_x, center_y))

    return square_centers


def plot_intersections(domain_width, domain_height, intersections):
    """
    Plot the trajectory intersections as diagonal streaks, excluding horizontal and vertical streaks.

    Args:
    - domain_width: Width of the domain.
    - domain_height: Height of the domain.
    - intersections: List of (x, y) intersection points.
    """
    plt.figure(figsize=(12, 6))
    plt.xlim(0, domain_width)
    plt.ylim(0, domain_height)
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot the domain boundaries
    plt.plot([0, domain_width, domain_width, 0, 0],
             [0, 0, domain_height, domain_height, 0], 'k-', label="Domain Borders")

    # Draw streaks
    
    for i in range(1, len(intersections)):
        x1, y1 = intersections[i - 1]
        x2, y2 = intersections[i]

        # Plot only if the points do not share an axis value
        if x1 != x2 and y1 != y2:
            plt.plot([x1, x2], [y1, y2], '-r')

    # Plot intersection points
    plt.scatter(*zip(*intersections), color='blue', label="Intersection Points")
    plt.title("Diagonal Streaks (Skip Same-Axis Intersections)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def bundle_segments(points):
    """
    Bundle a sequence of 2D points into line segments starting from index 1.

    Args:
    - points: List of 2D points [(x, y)].

    Returns:
    - segments: List of line segments [(x1, y1, x2, y2)] starting from index 1.
    """
    segments = []
    for i in range(1, len(points),2):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        segments.append((x1, y1, x2, y2))
    return segments

def place_squares_trajectory(trajectory_segments, square_size, trajectory_angle, domain_height):
    """
    Place squares along the trajectory to ensure no overlap on the same line.

    Args:
    - trajectory_segments: List of line segments [(x1, y1, x2, y2)].
    - square_size: Size of the square.

    Returns:
    - square_centers: List of square centers [(x, y)].
    """
    square_centers = []
    x1, y1, x2, y2 = trajectory_segments[0]
    segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    direction_vector = ((x2 - x1) / segment_length, (y2 - y1) / segment_length)
    efective_length = (square_size)/(2*direction_vector[1])
    
    # print(direction_vector[1]/direction_vector[0])
    
    for j, segment in enumerate(trajectory_segments):
        x1, y1, x2, y2 = segment
        segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_squares = int(np.ceil((y2-y1) / square_size))

        if y2 >= domain_height:
            # breakpoint()
            y2 = math.ceil(domain_height/square_size)*square_size
            x2 += (direction_vector[0]/direction_vector[1])*(y2-domain_height)
            
        # Place square centers along the segment
        if y1==0:
            for i in range(0,num_squares):
                center_x = (x1+efective_length * direction_vector[0]) + (i *2*efective_length) * direction_vector[0]
                center_y = (y1+efective_length * direction_vector[1]) + (i * 2*efective_length) * direction_vector[1]
                if center_x>21600:
                    center_y -= (center_x-21600)*direction_vector[1]/direction_vector[0]
                    center_x = 21600
                if center_y>10800:
                    center_x -= (center_y-10800)*direction_vector[0]/direction_vector[1]
                    center_y = 10800
                square_centers.append((center_x, center_y))
        else:
            temp_centers = []
            for i in range(0,num_squares):
                center_x = (x2-efective_length * direction_vector[0]) - (i *2*efective_length) * direction_vector[0]
                center_y = (y2-efective_length * direction_vector[1]) - (i * 2*efective_length) * direction_vector[1]
                if center_x<0:
                    center_y += (-center_x)*direction_vector[1]/direction_vector[0]
                    center_x = 0
                if center_y<0:
                    center_x += (-center_y)*direction_vector[0]/direction_vector[1]
                    center_y = 0
                if center_y>10800:
                    center_x -= (center_y-10800)*direction_vector[0]/direction_vector[1]
                    center_y = 10800
                temp_centers.append((center_x, center_y))
            square_centers += temp_centers[::-1]
        # temp_centers = []
        # for ix, center in enumerate(square_centers):
        #     if ix>0 and center[0]-square_centers[ix-1][0] > 300:
        #         temp_centers.append(center)
        # square_centers = temp_centers

    return square_centers

def plot_squares(trajectory_segments, square_centers, square_size, domain_width, domain_height, start=None):
    """
    Plot the trajectory segments and squares along the trajectory, limiting the view to the domain.

    Args:
    - trajectory_segments: List of line segments [(x1, y1, x2, y2)].
    - square_centers: List of square centers [(x, y)].
    - square_size: Size of the square.
    - domain_width: Width of the domain.
    - domain_height: Height of the domain.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # Set domain limits
    ax.set_xlim(0, domain_width)
    ax.set_ylim(0, domain_height)

    # Plot trajectory segments
    for segment in trajectory_segments:
        x1, y1, x2, y2 = segment
        ax.plot([x1, x2], [y1, y2], 'k-', label='Trajectory' if segment == trajectory_segments[0] else '')

    if start:
        plt.plot(start[0],start[1],marker='x')
    # Plot squares
    size = square_size/2
    for cx, cy in square_centers:
        # Only plot squares within the domain boundaries
        
        square = plt.Rectangle((cx-size, cy -size), square_size, square_size,
                                edgecolor='blue', facecolor='none')
        ax.set_xlim(0, domain_width)
        ax.set_ylim(0, domain_height)
        ax.add_patch(square)
        ax.plot(cx,cy,marker='x',c='orange')
        # plt.pause(0.0001)

    plt.title("Squares Along Trajectory within Domain")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def adaptivempc(x0, y0, v_x, v_y, square_size,domain_width=21600, domain_height=10800, proximity_threshold=100):
    
    start_x = x0-10800*v_x/v_y
    start_x = start_x if start_x>0 else start_x+21600
    intersections = calculate_border_intersections(domain_width, domain_height, start_x, y0, v_x, v_y, proximity_threshold, square_size)
    
    trajectory = bundle_segments(intersections[1:])

    centers =   place_squares_trajectory(trajectory,square_size, v_y/v_x, domain_height)
    
    # plot_squares(trajectory,centers,square_size,domain_width, domain_height, start=[x0,y0])
    
    return centers, trajectory

if __name__ == '__main__':
    
    # Parameters
    domain_width = 21600
    domain_height = 10800
    x0 = 500
    y0=250
    v_x = 4.35  # Velocity in x-direction
    v_y = 5.49  # Velocity in y-direction
    height = 590

    declive = v_y/v_x
    angle = math.atan(declive)
    proximity_threshold = 100  # Distance threshold for stopping

    # Step 1: Calculate intersection points with proximity-based stopping
    intersections = calculate_border_intersections(domain_width, domain_height, x0, y0, v_x, v_y, proximity_threshold, height)
    # Step 2: Plot the trajectory intersections
    # plot_intersections(domain_width, domain_height, intersections)


    m_distance = find_distance_between_trajectories(intersections)
    print(m_distance)
    
    intersections = calculate_border_intersections(domain_width, domain_height, x0+m_distance, y0, v_x, v_y, proximity_threshold, height)
    m_distance = find_distance_between_trajectories(intersections)
    
    midpoint = find_midpoint_of_trajectory(intersections)

    closest_value = map_to_closest_value(m_distance/2, [500, 400, 300])
    trajectory = bundle_segments(intersections[1:])
    # Step 3: Create a grid centered at the midpoint with squares twice the mapped size
    square_size = 2 * closest_value
    centers = place_squares_trajectory(trajectory,height, declive, domain_height)

    plot_squares(trajectory,centers,height,domain_width, domain_height,[x0,y0])