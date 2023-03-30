import datetime
import numpy as np

LEMD_LATITUDE = 40.49
LEMD_LONGITUDE = -3.585

LEMD_LATITUDE = 40.4922683
LEMD_LONGITUDE = -3.5629642

def haversine_np(lat1: np.array, lon1: np.array, lat2: np.array = LEMD_LATITUDE,
                 lon2: np.array = LEMD_LONGITUDE, h1: np.array = None, h2: np.array = None, 
                 angle: bool = False) -> np.array:
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """Calculate distances between succesive GPS positions.

    Calculate the great circle distance between segments on the earth. 
    Input can be provided as arrays of coordinates.
    All args must be of equal length.
    
    Args:
        lat1, lon1: Latitude and longitude coordinates of initial GPS points.
        lat2, lon2: Latitude and longitude coordinates of final GPS points.
        h1, h2: Initial and final altitude values (optional)
        angle: Determines if direction momentum is introduced, and function acts as
            a cost function directly proportional to angle variation
    """    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))  
    mi = 3959 * c
    
    if h1 is not None:
        mi = np.sqrt(np.power(mi,2) + np.power((h2-h1)/5280, 2)) 
        
    if angle:
        vectors = np.array([dlon,dlat]).transpose()
        
        p1 = np.einsum('ij,ij->i',vectors[1:],vectors[:-1])
        p2 = np.einsum('ij,ij->i',vectors[1:],vectors[1:])
        p3 = np.einsum('ij,ij->i',vectors[:-1],vectors[:-1])
        
        p4 = p1 / (np.sqrt(p2*p3))
        angles = np.abs(np.arccos(np.clip(p4,-1.0,1.0)))
        
        mi2 = mi * np.exp( np.concatenate([[0],angles])/2)

        return mi2
    return mi

def haversine_np_track(lat1: np.array, lon1: np.array, lat2: np.array = LEMD_LATITUDE,
                 lon2: np.array = LEMD_LONGITUDE, track1: np.array = None, track2: np.array = None, 
                 angle: bool = False) -> np.array:
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """Calculate distances between succesive GPS positions.

    Calculate the great circle distance between segments on the earth. 
    Input can be provided as arrays of coordinates.
    All args must be of equal length.
    
    Args:
        lat1, lon1: Latitude and longitude coordinates of initial GPS points.
        lat2, lon2: Latitude and longitude coordinates of final GPS points.
        h1, h2: Initial and final altitude values (optional)
        angle: Determines if direction momentum is introduced, and function acts as
            a cost function directly proportional to angle variation
    """    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))  
    mi = 3959 * c 
        
    if angle:
        dtrack = track2 - track1
        dtrack[np.abs(dtrack)>=180] = (dtrack[np.abs(dtrack)>=180] - np.sign(dtrack[np.abs(dtrack)>=180])*360)
        dtrack = np.absolute(dtrack).astype('float')
        # mi = mi * np.exp( np.radians(dtrack))
        mi = mi * (1 + 5*dtrack)
    return mi


def get_dates_between(date_start: str, date_end: str) -> list[datetime.datetime]:
    """Retrieves a list of datetime objects between two dates.

    Args:
        date_start: Start date in YYYY-MM-DD format 
        date_end: End date in YYYY-MM-DD format 

    """
    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_end   = datetime.datetime.strptime(date_end,   '%Y-%m-%d')
    dates      = [(date_start + datetime.timedelta(days=x))
                  for x in range((date_end - date_start).days + 1)]
    return dates
