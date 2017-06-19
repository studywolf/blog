'''
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import scipy.optimize


class Arm3Link:

    def __init__(self, q=None, q0=None, L=None):
        """Set up the basic parameters of the arm.
        All lists are in order [shoulder, elbow, wrist].

        q : np.array
            the initial joint angles of the arm
        q0 : np.array
            the default (resting state) joint configuration
        L : np.array
            the arm segment lengths
        """
        # initial joint angles
        self.q = [.3, .3, 0] if q is None else q
        # some default arm positions
        self.q0 = np.array([np.pi/4, np.pi/4, np.pi/4]) if q0 is None else q0
        # arm segment lengths
        self.L = np.array([1, 1, 1]) if L is None else L

        self.max_angles = [np.pi, np.pi, np.pi/4]
        self.min_angles = [0, 0, -np.pi/4]

    def get_xy(self, q=None):
        """Returns the corresponding hand xy coordinates for
        a given set of joint angle values [shoulder, elbow, wrist],
        and the above defined arm segment lengths, L

        q : np.array
            the list of current joint angles

        returns : list
            the [x,y] position of the arm
        """
        if q is None:
            q = self.q

        x = self.L[0]*np.cos(q[0]) + \
            self.L[1]*np.cos(q[0]+q[1]) + \
            self.L[2]*np.cos(np.sum(q))

        y = self.L[0]*np.sin(q[0]) + \
            self.L[1]*np.sin(q[0]+q[1]) + \
            self.L[2]*np.sin(np.sum(q))

        return [x, y]

    def inv_kin(self, xy):
        """This is just a quick write up to find the inverse kinematics
        for a 3-link arm, using the SciPy optimize package minimization
        function.

        Given an (x,y) position of the hand, return a set of joint angles (q)
        using constraint based minimization, constraint is to match hand (x,y),
        minimize the distance of each joint from it's default position (q0).

        xy : tuple
            the desired xy position of the arm

        returns : list
            the optimal [shoulder, elbow, wrist] angle configuration
        """

        def distance_to_default(q, *args):
            """Objective function to minimize
            Calculates the euclidean distance through joint space to the
            default arm configuration. The weight list allows the penalty of
            each joint being away from the resting position to be scaled
            differently, such that the arm tries to stay closer to resting
            state more for higher weighted joints than those with a lower
            weight.

            q : np.array
                the list of current joint angles

            returns : scalar
                euclidean distance to the default arm position
            """
            # weights found with trial and error,
            # get some wrist bend, but not much
            weight = [1, 1, 1.3]
            return np.sqrt(np.sum([(qi - q0i)**2 * wi
                           for qi, q0i, wi in zip(q, self.q0, weight)]))

        def x_constraint(q, xy):
            """Returns the corresponding hand xy coordinates for
            a given set of joint angle values [shoulder, elbow, wrist],
            and the above defined arm segment lengths, L

            q : np.array
                the list of current joint angles
            xy : np.array
                current xy position (not used)

            returns : np.array
                the difference between current and desired x position
            """
            x = (self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1]) +
                 self.L[2]*np.cos(np.sum(q))) - xy[0]
            return x

        def y_constraint(q, xy):
            """Returns the corresponding hand xy coordinates for
            a given set of joint angle values [shoulder, elbow, wrist],
            and the above defined arm segment lengths, L

            q : np.array
                the list of current joint angles
            xy : np.array
                current xy position (not used)
            returns : np.array
                the difference between current and desired y position
            """
            y = (self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1]) +
                 self.L[2]*np.sin(np.sum(q))) - xy[1]
            return y

        def joint_limits_upper_constraint(q, xy):
            """Used in the function minimization such that the output from
            this function must be greater than 0 to be successfully passed.

            q : np.array
                the current joint angles
            xy : np.array
                current xy position (not used)

            returns : np.array
                all > 0 if constraint matched
            """
            return self.max_angles - q

        def joint_limits_lower_constraint(q, xy):
            """Used in the function minimization such that the output from
            this function must be greater than 0 to be successfully passed.

            q : np.array
                the current joint angles
            xy : np.array
                current xy position (not used)

            returns : np.array
                all > 0 if constraint matched
            """
            return q - self.min_angles

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.q,
            eqcons=[x_constraint,
                    y_constraint],
            # uncomment to add in min / max angles for the joints
            # ieqcons=[joint_limits_upper_constraint,
            #          joint_limits_lower_constraint],
            args=(xy,),
            iprint=0)  # iprint=0 suppresses output


def test():
    # ###########Test it!##################

    arm = Arm3Link()

    # set of desired (x,y) hand positions
    x = np.arange(-.75, .75, .05)
    y = np.arange(.25, .75, .05)

    # threshold for printing out information, to find trouble spots
    thresh = .025

    count = 0
    total_error = 0
    # test it across the range of specified x and y values
    for xi in range(len(x)):
        for yi in range(len(y)):
            # test the inv_kin function on a range of different targets
            xy = [x[xi], y[yi]]
            # run the inv_kin function, get the optimal joint angles
            q = arm.inv_kin(xy=xy)
            # find the (x,y) position of the hand given these angles
            actual_xy = arm.get_xy(q)
            # calculate the root squared error
            error = np.sqrt(np.sum((np.array(xy) - np.array(actual_xy))**2))
            # total the error
            total_error += np.nan_to_num(error)

            # if the error was high, print out more information
            if np.sum(error) > thresh:
                print('-------------------------')
                print('Initial joint angles', arm.q)
                print('Final joint angles: ', q)
                print('Desired hand position: ', xy)
                print('Actual hand position: ', actual_xy)
                print('Error: ', error)
                print('-------------------------')

            count += 1

    print('\n---------Results---------')
    print('Total number of trials: ', count)
    print('Total error: ', total_error)
    print('-------------------------')

if __name__ == '__main__':
    test()
