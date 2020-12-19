from joint_intent_slot_filling.location_extractor import LocationExtractor
import joint_intent_slot_filling.constants as consts
import pandas as pd


class LocationExtractorTest:
    def __init__(self, all_locations,
                 test_data_path=consts.LOCATION_EXTRACTOR_TEST_DATA_PATH,
                 test_metrics_path=consts.LOCATION_EXTRACTOR_TEST_RESULT_METRICS):
        self.location_extractor = LocationExtractor(all_locations=all_locations)
        self.data = pd.read_csv(test_data_path)
        self.test_metrics_path = test_metrics_path

    def test(self):
        with open(self.test_metrics_path, "w") as f:
            # test levenshtein
            well_extracted = 0
            number_of_data = 0
            f.write("====== Levenshtein Distance Results ======\n")
            for index, row in self.data.iterrows():
                extracted_location = self.location_extractor.get_most_similar_location_levenshtein(row.predicted)
                if row.expected == extracted_location:
                    well_extracted += 1
                number_of_data += 1
                f.write("Predicted Value (Input): " + row.predicted + " -- Expected Value: " + row.expected +
                        " -- Extracted Value: " + str(extracted_location) + "\n")

            f.write("Percentage of correct extractions: " + str(round(float(well_extracted) / number_of_data * 100, 2)) + "%\n")
            f.write("=============================================\n\n\n")

            # test damerau_levenshtein
            well_extracted = 0
            number_of_data = 0
            f.write("====== Damerau-Levenshtein Distance Results ======\n")
            for index, row in self.data.iterrows():
                extracted_location = self.location_extractor.get_most_similar_location_damerau_levenshtein(row.predicted)
                if row.expected == extracted_location:
                    well_extracted += 1
                number_of_data += 1
                f.write("Predicted Value (Input): " + row.predicted + " -- Expected Value: " + row.expected +
                        " -- Extracted Value: " + str(extracted_location) + "\n")

            f.write("Percentage of correct extractions: " + str(round(float(well_extracted) / number_of_data * 100, 2)) + "%\n")
            f.write("=============================================\n\n\n")

            # test smith_waterman
            well_extracted = 0
            number_of_data = 0
            f.write("====== Smith Waterman Distance Results ======\n")
            for index, row in self.data.iterrows():
                extracted_location = self.location_extractor.get_most_similar_location_smith_waterman(row.predicted)
                if row.expected == extracted_location:
                    well_extracted += 1
                number_of_data += 1
                f.write("Predicted Value (Input): " + row.predicted + " -- Expected Value: " + row.expected +
                        " -- Extracted Value: " + str(extracted_location) + "\n")

            f.write("Percentage of correct extractions: " + str(round(float(well_extracted) / number_of_data * 100, 2)) + "%\n")
            f.write("=============================================\n\n\n")

            # test sequence_matcher
            well_extracted = 0
            number_of_data = 0
            f.write("====== Sequence Matcher Distance Results ======\n")
            for index, row in self.data.iterrows():
                extracted_location = self.location_extractor.get_most_similar_location_sequence_matcher(row.predicted)
                if row.expected == extracted_location:
                    well_extracted += 1
                number_of_data += 1
                f.write("Predicted Value (Input): " + row.predicted + " -- Expected Value: " + row.expected +
                        " -- Extracted Value: " + str(extracted_location) + "\n")

            f.write("Percentage of correct extractions: " + str(round(float(well_extracted) / number_of_data * 100, 2)) + "%\n")
            f.write("=============================================\n\n\n")


if __name__ == "__main__":
    locations = ['Deep Sea Dungeon',
                 'Castle Of Doom Dungeon',
                 'Dragon Dungeon',
                 'Tutorial Dungeon',
                 'Teleporter',
                 'Healing Shrine',
                 'Weapon Shop',
                 'Armor Shop',
                 'Potion Shop']
    location_extractor_test = LocationExtractorTest(all_locations=locations)
    location_extractor_test.test()
