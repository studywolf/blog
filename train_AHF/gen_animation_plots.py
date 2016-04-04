'''
Copyright (C) 2016 Travis DeWolf

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
import matplotlib.pyplot as plt
import glob
import sys

from plot_error import gen_data_plot 

folder = "weights" if len(sys.argv) < 2 else sys.argv[1]
total_num = 100 if len(sys.argv) < 3 else int(sys.argv[2])

files = sorted(glob.glob('%s/rnn*'%folder))
step = len(files) / total_num
count = 0
for ii in range(1, len(files), step):
    name = '%s/plots/%05i.png'%(folder, count)
    print 'generating plot %i...'%count
    gen_data_plot(folder, ii, show_plot=False, verbose=False, save_plot=name)
    count += 1
# now generate 10 more plots of the last frame so the 
# gif seems to pause a bit at the end
for jj in range(10):
    name = '%s/plots/%05i.png'%(folder, count)
    print 'generating plot %i...'%count
    gen_data_plot(folder, ii, show_plot=False, verbose=False, save_plot=name)
    count += 1
