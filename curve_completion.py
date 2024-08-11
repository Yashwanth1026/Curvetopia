from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import numpy as np

def complete_curve(points, occlusion_type='connected'):
    # Sort points by x to ensure monotonicity
    points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    y = points[:, 1]
    
    if occlusion_type == 'connected':
        # Use interpolation to complete the curve
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        new_x = np.linspace(x[0], x[-1], num=500)
        new_y = f(new_x)
        completed_points = np.column_stack((new_x, new_y))
    elif occlusion_type == 'disconnected':
        # Handle disconnected occlusions by connecting endpoints
        completed_points = handle_disconnected_occlusions(points)
        
    return completed_points

def handle_disconnected_occlusions(points):
    """
    Handle disconnected occlusions by connecting the endpoints of fragments.
    """
    if len(points) < 2:
        return points
    
    # Find start and end points
    start_point = points[0]
    end_point = points[-1]
    
    # Create a line connecting the start and end points
    lin_reg = LinearRegression()
    lin_reg.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    new_x = np.linspace(start_point[0], end_point[0], num=500)
    new_y = lin_reg.predict(new_x.reshape(-1, 1))
    
    # Concatenate original points with the new points
    completed_points = np.vstack((points, np.column_stack((new_x, new_y))))
    return completed_points
