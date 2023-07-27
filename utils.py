import math
from collections import OrderedDict
import re,math


def xyz_list2dict(l):
    return OrderedDict({'x':l[0], 'y':l[1], 'z':l[2]})


def deg2rad(expr):
    """Convert value expression to radians unit"""
    # Use regular expression to find all numeric values in the expression string
    numeric_values = re.findall(r'[+-]?\d+(?:\.\d+)?', expr)

    # Convert each numeric value to radians and sum them up
    radians = sum(math.radians(float(value)) for value in numeric_values)
    return radians

def find_numeric(expr):
    """
    Find a single numeric value from the string expression
    
    """
    numeric_value = float(re.search(r'[+-]?\d+(?:\.\d+)?', expr).group())
    return numeric_value

def angle_from_vector_to_x(vec):
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle
