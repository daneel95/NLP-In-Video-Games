# In this file we define a class that will have the similarity functions.
# Its usage will be: instantiation of the class then just calling the wanted function from it
# The reason for putting them in the same place is so it's easier to handle it in a single place.
# Ideally, they should be split, but no reason for such a small project

from textdistance import levenshtein, damerau_levenshtein, smith_waterman
import difflib


class LocationExtractor:
    def __init__(self, all_locations):
        self.all_locations = all_locations

    # Calculate levenshtein similarity (how many changes are required to get from one string to another)
    # The location with lowest number of required changes is the one that will be chosen.
    def get_most_similar_location_levenshtein(self, location, threshold=8):
        normalized = self.__normalize_string(location)
        best_location = ""
        min_similarity = 99999999999
        for loc in self.all_locations:
            normalized_loc = self.__normalize_string(loc)
            similarity = levenshtein(normalized, normalized_loc)
            if similarity < min_similarity:
                min_similarity = similarity
                best_location = loc
        if min_similarity > threshold:
            return None
        return best_location

    # Calculate damerau levenshtein similarity. It is the same as Levenstein similarity but also includes
    # two-character transpositions.
    def get_most_similar_location_damerau_levenshtein(self, location, threshold=8):
        normalized = self.__normalize_string(location)
        best_location = ""
        min_similarity = 99999999999
        for loc in self.all_locations:
            normalized_loc = self.__normalize_string(loc)
            similarity = damerau_levenshtein(normalized, normalized_loc)
            if similarity < min_similarity:
                min_similarity = similarity
                best_location = loc
        if min_similarity > threshold:
            return None
        return best_location

    # Calculate smith waterman similarity
    def get_most_similar_location_smith_waterman(self, location, threshold=1):
        normalized = self.__normalize_string(location)
        best_location = ""
        max_similarity = -1
        for loc in self.all_locations:
            normalized_loc = self.__normalize_string(loc)
            similarity = smith_waterman(normalized, normalized_loc)
            if similarity > max_similarity:
                max_similarity = similarity
                best_location = loc
        # nothing is similar, something is fishy
        if max_similarity < threshold:
            return None
        return best_location

    # Calculate similarity using sequences matcher
    def get_most_similar_location_sequence_matcher(self, location, threshold=0.25):
        normalized = self.__normalize_string(location)
        best_location = ""
        max_similarity = -1
        for loc in self.all_locations:
            normalized_loc = self.__normalize_string(loc)
            similarity = difflib.SequenceMatcher(isjunk=None, a=normalized, b=normalized_loc).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                best_location = loc
        # nothing is similar, something is fishy
        if max_similarity < threshold:
            return None
        return best_location

    def __normalize_string(self, str_value):
        return str_value.lower().replace(" ", "")
