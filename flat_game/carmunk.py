import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = False
draw_screen = False


def draw(screen, space):
    space.debug_draw(DrawOptions(screen))

class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        #veloicty contraints
        self.velocity_max = 1000
        self.velocity_min = -50
        self.velocity_init = 100
        self.velocity = self.velocity_init

        self.x_last = 0
        self.y_last = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200, 125))
        self.obstacles.append(self.create_obstacle(600, 600, 35))

        # Create cats.
        self.cats = [self.create_cat() for _ in range(3)]

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_shape = pymunk.Circle(c_body, r)
        #c_shape = pymunk.Poly(c_body,[(5,5), (-5,5), (0,10)], radius =r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        cat = {
            "body" : pymunk.Body(1, inertia)
        }
        cat['shape'] = pymunk.Circle(cat['body'], 30)
        cat['body'].position = 50, height - 100
        cat['shape'].color = THECOLORS["orange"]
        cat['shape'].elasticity = 1.0
        cat['shape'].angle = 0.5
        direction = Vec2d(1, 0).rotated(cat['body'].angle)
        self.space.add(cat['body'], cat['shape'])

        return cat

    def create_car(self, x, y, r):
        print ('creating car')
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        # self.car_body.apply_impulse(driving_direction)
        # modify this for local
        self.car_body.apply_impulse_at_world_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

        print('finished creating car')

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .2

        elif action == 1:  # Turn right.
            self.car_body.angle += .2

        elif action == 2:  # accelerate.
            self.velocity += 10
            if self.velocity > self.velocity_max:
                self.velocity = self.velocity_max

        elif action == 3:  # decelerate.
            self.velocity -= 10
            if self.velocity > self.velocity_max:
                self.velocity = self.velocity_max

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = self.velocity * driving_direction

        # advance the space
        self.space.step(1./10)
        clock.tick()

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        draw(screen, self.space)

        if draw_screen:
            pygame.display.flip()


        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        state = np.array([readings])

        #create a rewards based on distance moved
        distance_traveled = math.sqrt(math.pow(self.x_last - x, 2) + \
                            math.pow(self.y_last - y, 2))

        self.x_last = x
        self.y_last = y

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -2500
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = distance_traveled
        self.num_steps += 1


        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        for cat in self.cats:
            speed = random.randint(20, 200)
            cat['body'].angle -= random.randint(-1, 1)
            direction = Vec2d(1, 0).rotated(cat['body'].angle)
            cat['body'].velocity = speed * direction

    def car_is_crashed(self, readings):
        if len([x for x in readings if x == 1]):
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go reverse of current direction.
            epsilon = .01 # to avoid division by zero error
            self.car_body.velocity = -((self.velocity + epsilon)/(self.velocity + epsilon)) * \
                                     self.velocity_init * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["red"])  # Red is scary!
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

            self.velocity = self.velocity_init

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arms = [(self.make_sonar_arm(x, y), i) for i in [0,
            math.pi/4, -math.pi/4, math.pi/2, -math.pi/2,
            3*math.pi/4, -3*math.pi/4, math.pi]]

        # arms = [(self.make_sonar_arm(x, y), i) for i in [0]]


        # Rotate them and get readings.
        for arm, a in arms:

            distance = self.get_arm_distance(arm, x, y, angle, a)
            readings.append(distance)


        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):

        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)

        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black'] or reading == THECOLORS['green']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
