"""
cmdline drivers for unipen data set
"""
import cPickle
import os.path
import sys
import numpy as np
from dataset import DatFile

def read_file(input_name, output_name, box, to_array=True, to_csv=False):

    dat = DatFile()
    dat.read(open(input_name, 'rb'))

    segments = ['nan, nan']
    num_sequences = 5

    x_scale = (box[1] - box[0]) / float(num_sequences)
    y_scale = box[3] - box[2]

    for jj in range(num_sequences):
        # get sequence
        npa = np.array(dat.pen_downs[jj])
        # if there's no data skip it
        if len(npa) < 2:
            continue
        # for normalization
        xmin = npa[:, 0].min()
        xmax = npa[:, 0].max()
        ymin = npa[:, 1].min()
        ymax = npa[:, 1].max()

        for ii in range(len(npa)):
            x0, y0 = npa[ii]
            segments.append("%s, %s" % (
                    jj*x_scale + 
                    (x_scale * (x0 - xmin) / (xmax - xmin)) + box[0],
                    box[3] - y_scale * (ymax - y0) / (ymax - ymin)))

        segments.append('nan, nan')
    
    if to_csv:
        open(output_name, 'wb').write('\n'.join(segments))
    if to_array:
        array = np.zeros((len(segments), 2), dtype='float32')
        for i in range(len(segments)):
            row = segments[i].split(',')
            array[i] = row
        return array


def read_file_pkl(input_name, output_name, box, to_array=True, to_csv=False):
    pen_pos = cPickle.load(open(input_name, 'rb'))
    xmin = pen_pos[:, 0].min()
    xmax = pen_pos[:, 0].max()
    ymin = pen_pos[:, 1].min()
    ymax = pen_pos[:, 1].max()

    pen_pos[:, 0] -= xmin
    pen_pos[:, 0] /= (xmax - xmin)
    pen_pos[:, 1] -= ymin
    pen_pos[:, 1] /= (ymax - ymin)

    pen_pos[:, 0] *= box[1] - box[0]
    pen_pos[:, 0] += box[0]
    pen_pos[:, 1] *= box[3] - box[2]
    pen_pos[:, 1] += box[2]

    xmin = pen_pos[:, 0].min()
    xmax = pen_pos[:, 0].max()
    ymin = pen_pos[:, 1].min()
    ymax = pen_pos[:, 1].max()

    # wrap in nans
    nans = (np.ones((1,2))*np.nan)
    pen_pos = np.vstack([nans, pen_pos, nans])

    return pen_pos
