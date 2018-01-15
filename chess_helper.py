import chess
"""
This file is responsible for all chess logic that isn't Stockfish.
"""

class chess_helper:

    RIVAL = False
    ME = True

    def __init__(self, start_player):
        self.board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.curr_player = start_player
        if(True):
            self.user_starts = True
        else:
            self.user_starts = False

    """
    updates the board.
    :return whether turn is legal
    """
    def do_turn(self, src, dest):
        stringmove = src+dest
        move2do = chess.Move.from_uci(stringmove)
        if (move2do not in self.board.legal_moves):
            return False
        self.board.push(move2do)
        if(self.curr_player == chess_helper.ME):
            self.curr_player = chess_helper.RIVAL
        else:
            self.curr_player = chess_helper.ME
        return True

    """
        :src source square
        :return all destinations that can be reached from square.
        (uci format)
    """
    def square_dests(self, src):
        dests = []
        for move in self.board.legal_moves:
            if move.uci()[0:2] == src:
                dests.append(move.uci()[2:4])
        return dests

    """
        :dest destination square
        :return all sources that can reach square.
        (uci format)
    """
    def square_srcs(self, dest):
        srcs = []
        for move in self.board.legal_moves:
            if move.uci()[2:4] == dest:
                srcs.append(move.uci()[0:2])
        return srcs

    def square_color(self, square_location):
        letter = square_location[0]
        number = int(square_location[1]) - 1
        numLine = ord(letter) - 97
        is_white = ((numLine+number)%2==1)
        if is_white:
            return chess.WHITE
        return chess.BLACK

    def piece_color(self, square_location):
        letter = square_location[0]
        number = int(square_location[1]) - 1
        numLine = ord(letter) - 97
        square = number * 8 + numLine
        piece = self.board.piece_at(square)
        if piece == None:
            return None
        if piece.color == chess.WHITE:
            return chess.WHITE
        if piece.color == chess.BLACK:
            return chess.BLACK

    """
        :return all squares that a move can start from.
    """
    def get_sources(self):
        sources = []
        for move in self.board.legal_moves:
            if not move.uci()[0:2] in sources:
                sources.append(move.uci()[0:2])
        return sources

    def get_destinations(self):
        dests = []
        for move in self.board.legal_moves:
            if not move.uci()[2:4] in dests:
                dests.append(move.uci()[2:4])
        return dests

    """
    converts uci format to indexes in array. ZERO-BASED
    user_start - whether the user is white
    """
    def ucitoidx(self, str):
        x = ord(str[0]) - ord('a')
        y = int(str[1])
        if (not self.user_starts):
            y = 8 - y
        return [x, y]

    def get_relevant_locations(self):
        sources = self.get_sources()
        dests = self.get_destinations()
        return [sources, dests]


    def get_current_player(self):
        return self.curr_player