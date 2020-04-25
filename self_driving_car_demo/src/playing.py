"""
Once a model is trained, use this to play it.
"""
# third-party
import numpy
# first-party
import carmunk
import nn

NUM_SENSORS = 3

def play(model):
    """
    DOCSTRING
    """
    car_distance = 0
    game_state = carmunk.GameState()
    _, state = game_state.frame_step((2))
    while True:
        car_distance += 1
        action = (numpy.argmax(model.predict(state, batch_size=1)))
        _, state = game_state.frame_step(action)
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

if __name__ == '__main__':
    saved_model = 'saved-models/164-150-100-50000-25000.h5'
    model = nn.Denseneural_net(NUM_SENSORS, [164, 150], saved_model)
    play(model)
