#!/usr/bin/env python
###############################################################################
#
# set_test.py - Description!
#
###############################################################################
# #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or #
# (at your option) any later version. #
# #
# This program is distributed in the hope that it will be useful, #
# but WITHOUT ANY WARRANTY; without even the implied warranty of #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the #
# GNU General Public License for more details. #
# #
# You should have received a copy of the GNU General Public License #
# along with this program. If not, see <http://www.gnu.org/licenses/>. #
# #
###############################################################################

__author__ = "Miriam Shiffman"
__copyright__ = "Copyright 2015"
__credits__ = ["Miriam Shiffman"]
__license__ = "GPL3"
__version__ = "0.0.1"
__maintainer__ = "Miriam Shiffman"
__email__ = ""
__status__ = "Development"

###############################################################################

#import code
import random
import itertools

###############################################################################
###############################################################################
###############################################################################
###############################################################################


class game:
    def __init__(self, cards=None):
        if cards is None:
            self.cards = [self.random_card() for _ in xrange(12)]
        else:
            self.cards = cards

    def __repr__(self):
        return '\n'.join(str(card) for card in self.cards)

    def random_card(self):
        """Return random tuple (color, shape, number)"""
        return tuple( random.choice([1,2,3]) for _ in xrange(4) )

    def is_set(self, x, y, z):
        """Returns true if card trio is a set at every position based on sum mod 3, which is only 0 if all same or all different"""
        return sum( sum(i)%3 for i in zip(x,y,z) ) == 0

    def compare_all(self):
        self.trios = itertools.combinations( enumerate(self.cards), 3 )
        for x, y, z in self.trios:
            if self.is_set(x[1], y[1], z[1]):
                return (x[0], y[0], z[0])
        return None

    def play(self, prnt=False):
        if self.compare_all():
            res = [self.cards[i] for i in self.compare_all()]
            if prnt:
                print res
            return res
        return False

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def test(n):
    return '{}%'.format(sum( game().play() for _ in xrange(n) )/float(n) * 100.0)

"""
# debug code in place with interactive python console

code.interact(local=locals())
"""


##############################################################################
###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':
    pass

###############################################################################
###############################################################################
###############################################################################
###############################################################################
