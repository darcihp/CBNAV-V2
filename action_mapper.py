'''
ACTION_SIZE = 11


def map_action(action):
    if action == 0:
        angular = 1.25
        linear = 0.1

    elif action == 1:
        angular = 1.0
        linear = 0.2

    elif action == 2:
        angular = 0.75
        linear = 0.3

    elif action == 3:
        angular = 0.5
        linear = 0.5

    elif action == 4:
        angular = 0.25
        linear = 0.75

    elif action == 5:
        angular = 0.0
        linear = 1.0

    elif action == 6:
        angular = -0.25
        linear = 0.75

    elif action == 7:
        angular = -0.5
        linear = 0.5

    elif action == 8:
        angular = -0.75
        linear = 0.3

    elif action == 9:
        angular = -1.00
        linear = 0.2

    elif action == 10:
        angular = -1.25
        linear = 0.1

    #DEBUG ONLY
    elif action == 11:
        angular = 0.0
        linear = 0.0
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return linear, angular
'''

'''
ACTION_SIZE = 3

def map_action(action):
    if action == 0:
        angular = 2
        linear = 0.0

    elif action == 1:
        angular = 0.0
        linear = 1

    elif action == 2:
        angular = -2
        linear = 0.0
        #DEBUG ONLY
    elif action == 11:
        angular = 0.0
        linear = 0.0

    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return linear, angular
'''

ACTION_SIZE = 7

def map_action(action):
    if action == 0:
        angular = 1.25
        linear = 0.3

    elif action == 1:
        angular = 1.0
        linear = 0.4

    elif action == 2:
        angular = 0.5
        linear = 0.5

    elif action == 3:
        angular = 0.0
        linear = 0.6

    elif action == 4:
        angular = -0.5
        linear = 0.5

    elif action == 5:
        angular = -1.0
        linear = 0.4
        
    elif action == 6:
        angular = -1.25
        linear = 0.3
    #DEBUG ONLY
    elif action == 11:
        angular = 0.0
        linear = 0.0
        
    else:
        raise AttributeError("Invalid Action: {}".format(action))

    return linear, angular
