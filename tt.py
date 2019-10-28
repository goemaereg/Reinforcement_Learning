class Tester(object):
    """."""
    def __init__(self):
        super(Tester, self).__init__()
        self.input_shape = (6,2)

    def _decode(self, encoded_goal):
        """ Turns a 1D encoded goal into an input_shaped goal """
        # FOR NOW ONLY 2D GOALS (can't be arsed to find general formula .-.)
        row = encoded_goal // self.input_shape[1]
        col = encoded_goal - (self.input_shape[1]*row)
        return (row,col)

    def _encode(self, goal):
        """ Turns a goal into its 1D encoded version """
        # FOR NOW ONLY 2D GOALS (can't be arsed to find general formula .-.)
        return goal[0]*self.input_shape[1] + goal[1]

tt = Tester()
msg = (3,1)
encoded = tt._encode(msg)
decoded = tt._decode(encoded)
print("Message: {}\nEncoded: {}\nDecoded:{}".format(msg, encoded, decoded))
