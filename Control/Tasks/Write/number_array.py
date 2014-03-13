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

def get_raw_number(number, writebox):
    """Return the set of points for a number 0-9
    
    number int: the number to get and wrap
    writebox list: [min x, max x, min y, max y]
    """
 
    if number == 0:   
        num = np.array([6, 15, 4, 15, 2, 11, 2, 6, 4, 1, 7, 1, 9, 6, 9, 
                        11, 8, 13, 7, 15, 5, 15], dtype=float).reshape(-1,2)
    elif number == 1:
        num = np.array([5, 15, 5, 6, 5, 3, 5, 1, 5, 1], 
                        dtype=float).reshape(-1,2)
    elif number == 2:
        num = np.array([3, 8, 3, 9, 4, 10, 5, 10, 6, 10, 7, 9, 7, 8, 7, 
                        7, 6, 5, 5, 4, 4, 3, 3, 2, 3, 1, 4, 1, 5, 1, 
                        6, 1, 7, 1, 8, 1], dtype=float).reshape(-1,2)
    elif number == 3:
        num = np.array([5, 15, 6, 15, 7, 15, 8, 15, 9, 15, 10, 14, 11, 13,
                        11, 12, 11, 11, 11, 10, 10, 9, 9, 8, 8, 8, 5, 
                        8, 8, 8, 9, 8, 10, 7, 11, 6, 11, 5, 11,
                        3, 10, 2, 9, 1, 8, 1, 7, 1, 6, 1, 5, 1, 5, 1], 
                        dtype=float).reshape(-1,2)
    elif number == 4:
        num = np.array([0, 15, 0, 10, 4.75, 10, 5, 10, 5, 15, 5, 15, 5, 0, 
                        5, 0], dtype=float).reshape(-1,2)
    elif number == 5:
        num = np.array([11, 15, 9, 15, 7, 15, 5, 15, 5, 13, 5, 11, 5, 10, 
                        7, 10, 9, 10, 10, 9, 11, 8, 11, 6, 11, 4, 11, 3, 
                        10, 2, 9, 1, 7, 1, 5, 1, 3, 2, 3, 2], 
                        dtype=float).reshape(-1,2)
    elif number == 6:
        num = np.array([9, 15, 8, 14, 7, 13, 6, 12, 5, 11, 4, 9, 4, 7, 4, 
                        6, 4, 5, 4, 3, 5, 2, 6, 1, 7, 1, 8, 1, 9, 1, 10, 
                        2 , 11, 3, 11, 4, 11, 5, 10, 6, 9, 7, 8, 7, 7, 7, 
                        6, 6, 4, 5, 4, 5], dtype=float).reshape(-1,2)
    elif number == 7:
        num = np.array([5, 15, 7, 15, 9, 15, 11, 15, 12, 15, 12, 14, 12, 12, 
                        11, 10, 10, 8, 9, 6, 8, 4, 7, 2, 7, 1, np.nan, np.nan,
                        5, 8, 12, 8], 
                        dtype=float).reshape(-1,2)
    elif number == 8:
        num = np.array([9, 15, 8, 15, 7, 15, 6, 14, 5, 13, 5, 10, 6, 9, 
                        7, 8, 9, 8, 10, 7, 11, 6, 11, 3, 10, 2, 9, 1, 7, 
                        1, 6, 2, 5, 3, 5, 6, 6, 7, 7, 8, 10, 9, 11, 10, 11, 13, 
                        10, 14, 9, 15, 8, 15], dtype=float).reshape(-1,2)
    elif number == 9:
        num = np.array([5, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 7, 11, 9, 11, 10, 11,
                        11, 11, 13, 10, 14, 9, 15, 7, 15, 6, 15, 5, 14, 4, 13, 
                        4, 11, 4, 10, 5, 9, 7, 8, 10, 8, 10, 8], 
                        dtype=float).reshape(-1,2)
    else:
        raise Exception('invalid request')

    # normalize dimensions to 1x1
    num[:,0] /= float(max(num[:,0]))
    num[:,1] /= float(max(num[:,1]))

    # center number
    num[:,0] -= .5 - (max(num[:,0]) - min(num[:,0])) / 2.0

    num[:,0] *= 5.0 / 6.0 * (writebox[1] - writebox[0]) 
    num[:,1] *= (writebox[3] - writebox[2])
    num[:,0] += writebox[0]
    num[:,1] += writebox[2]

    return num

def get_single_number(**kwargs):
    """Wrap the number with np.nans on either end
    """

    num = get_raw_number(**kwargs)
    new_array = np.zeros((num.shape[0]+2, num.shape[1]))
    new_array[0] = [np.nan, np.nan]
    new_array[-1] = [np.nan, np.nan]
    new_array[1:-1] = num

    return new_array

def get_sequence(sequence, writebox):
    """Returns a sequence of numbers

    sequence list: the sequence of integers
    writebox list: [min x, max x, min y, max y]
    """

    nans = np.array([np.nan, np.nan])
    nums= nans.copy()

    each_num_width = (writebox[1] - writebox[0]) / float(len(sequence))

    for ii, nn in enumerate(sequence):
        
        num_writebox = [writebox[0] + each_num_width * ii, 
                        writebox[0] + each_num_width * (ii+1), 
                        writebox[2], writebox[3]]

        num = get_raw_number(nn, num_writebox)
        nums = np.vstack([nums, num, nans])

    return nums 


### Testing code ###
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    nums = get_sequence([1,1,4], writebox=[-1,1,0,1])
    plt.plot(nums[:,0], nums[:,1])
    plt.show()
