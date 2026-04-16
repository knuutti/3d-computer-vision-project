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

def get_element(object: ObjectType) -> ElementType:
    if object in [ObjectType.CUBE_RED, ObjectType.CUBE_GREEN, ObjectType.CUBE_BLUE]:
        return ElementType.CUBE_FACE
    elif object in [ObjectType.GOAL_RED, ObjectType.GOAL_GREEN, ObjectType.GOAL_BLUE]:
        return ElementType.GOAL
    elif object == ObjectType.ROBOT_FRONT:
        return ElementType.ROBOT_FRONT
    elif object == ObjectType.ROBOT_CENTER:
        return ElementType.ROBOT_CENTER
    else:
        raise ValueError("Unknown object type")
    
def get_object_type(element: ElementType, color: int) -> ObjectType:
    if element == ElementType.CUBE_FACE:
        return ObjectType.CUBE_RED + color
    elif element == ElementType.GOAL:
        return ObjectType.GOAL_RED + color
    elif element == ElementType.ROBOT_FRONT:
        return ObjectType.ROBOT_FRONT
    elif element == ElementType.ROBOT_CENTER:
        return ObjectType.ROBOT_CENTER
    else:
        raise ValueError("Unknown element type")

def get_element_height(element: ElementType) -> float:
    if element == ElementType.CUBE_FACE:
        return 40.0
    elif element == ElementType.GOAL:
        return 0.0
    elif element == ElementType.ROBOT_FRONT:
        return 85.0
    elif element == ElementType.ROBOT_CENTER:
        return 90.0
    else:
        return 0.0

def get_element_length(element: ElementType) -> float:
    if element == ElementType.CUBE_FACE:
        return 20.0
    elif element == ElementType.GOAL:
        return 20.0
    elif element == ElementType.ROBOT_FRONT:
        return 60.0 # might not be accurate, but this is not used anyways :D
    elif element == ElementType.ROBOT_CENTER:
        return 120.0
    else:
        return 0.0