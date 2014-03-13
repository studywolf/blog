#!/usr/bin/env python
"""

Download from
http://www.ai.rug.nl/~lambert/overslag/unipen-CDROM-train_r01_v07.tgz
"""


def blocks(line_iter):
    """Iterate over the blocks of UNIPEN's .dat files
    """
    # XXX implement the include statement 
    line = line_iter.next()
    lines = [line]
    while True:
        line = line_iter.next()
        if line.startswith('.'):
            yield lines
            lines = []
        lines.append(line)


def pen_block(block):
    """Convert a PEN_UP or PEN_DOWN block into a sequence of (x, y) coords
    """
    assert len(block[0].split()) == 1
    locs = []
    for line in block[1:]:
        if len(line.strip()) == 0:
            continue
        try:
            x, y = line.split()
        except:
            raise
        locs.append((float(x), float(y)))
    return locs


class DatFile(object):
    """Simple reader for UNIPEN .dat file
    """
    def __init__(self):
        self.pen_ups = []
        self.pen_downs = []
        self.points_per_second = 100.0 # XXX READ full .dat file

    def block_COMMENT(self, block):
        pass

    def block_PEN_UP(self, block):
        self.pen_ups.append(pen_block(block))

    def block_PEN_DOWN(self, block):
        self.pen_downs.append(pen_block(block))

    def block_WRITER_INFO(self, block):
        pass

    def block_SEX(self, block):
        pass

    def block_AGE(self, block):
        pass

    def block_COUNTRY(self, block):
        pass

    def block_SKILL(self, block):
        pass

    def block_HAND(self, block):
        pass

    def block_DATE(self, block):
        pass

    def block_STYLE(self, block):
        pass

    def block_WRITER_ID(self, block):
        pass

    def read(self, file):
        # XXX verify that last block is read
        for block in blocks(iter(file)):
            block_type = block[0].split()[0][1:]
            getattr(self, 'block_%s' % block_type)(block)

