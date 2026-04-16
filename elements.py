class ElementType:
    CUBE_FACE = 0
    GOAL = 1
    ROBOT_CENTER = 2
    ROBOT_FRONT = 3

class ObjectType:
    CUBE_RED = 0
    CUBE_GREEN = 1
    CUBE_BLUE = 2
    GOAL_RED = 3
    GOAL_GREEN = 4
    GOAL_BLUE = 5
    ROBOT_FRONT = 6
    ROBOT_CENTER = 7

def get_element(object) -> ObjectType:
    if object.type == ElementType.CUBE_FACE:
        return ObjectType.CUBE_RED + object.color
    elif object.type == ElementType.GOAL:
        return ObjectType.GOAL_RED + object.color
    elif object.type == ElementType.ROBOT_FRONT:
        return ObjectType.ROBOT_FRONT
    elif object.type == ElementType.ROBOT_CENTER:
        return ObjectType.ROBOT_CENTER
    else:
        return None

def get_object_height(element) -> float:
    if element.type == ElementType.CUBE_FACE:
        return 40.0
    elif element.type == ElementType.GOAL:
        return 0.0
    elif element.type == ElementType.ROBOT_FRONT:
        return 85.0
    elif element.type == ElementType.ROBOT_CENTER:
        return 90.0
    else:
        return 0.0
    
def get_object_length(element) -> float:
    if element.type == ElementType.CUBE_FACE:
        return 20.0
    elif element.type == ElementType.GOAL:
        return 20.0
    elif element.type == ElementType.ROBOT_FRONT:
        return 60.0 # might not be accurate, but this is not used anyways :D
    elif element.type == ElementType.ROBOT_CENTER:
        return 120.0
    else:
        return 0.0