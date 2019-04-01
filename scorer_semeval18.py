# -*- coding: utf-8 -*-
from codecs import open
import sys

"""This script evaluates the systems on the SemEval 2018 task on Emoji Prediction.
   It takes the gold standard and system's output file as input and prints the results 
   in terms of macro and micro average F-Scores (0-100).
"""

class scorer_semeval18:
    def __init__(self, gold_key_path: str, output_path: str):
        """This initial the scorer that used formular provided by the contest organizer
        This code is replicated of the macro score provided by SemEval18 task 2 organizer
        """
        self.truth_dict = {}
        # For each top emoji, we will store the number our model predict it correctly in output_dict_correct
        # For each emojim, we will also store the total number of predict of that emoji from our model
        # These two variables are used to calculate individual F1 score and also for the macro-F1
        self.output_dict_correct = {}
        self.output_dict_attempted = {}
        truth_file_lines = open(path_goldstandard, encoding='utf8').readlines()
        submission_file_lines = open(path_outputfile, encoding='utf8').readlines()
        # The number of lines between output file and gold key have to be the same.
        if len(submission_file_lines) != len(truth_file_lines): sys.exit('ERROR: Number of lines in gold and output files differ')
        # populating the dictionary one entry for each emoji that appears in our gold_key and output from the model
        for i in range(len(submission_file_lines)):
            line = submission_file_lines[i]
            emoji_code_gold = truth_file_lines[i].replace("\n","")
            if emoji_code_gold not in self.truth_dict: self.truth_dict[emoji_code_gold] = 1
            else: self.truth_dict[emoji_code_gold] += 1
            emoji_code_output = submission_file_lines[i].replace("\n","")
            if emoji_code_output == emoji_code_gold:
                # Update the output_dict_correct each time we have the correct answer
                if emoji_code_output not in self.output_dict_correct: self.output_dict_correct[emoji_code_gold] = 1
                else: self.output_dict_correct[emoji_code_output] += 1
            # Update the output_dict_attempt, for each model prediction no matter right or wrong.
            if emoji_code_output not in self.output_dict_attempted: self.output_dict_attempted[emoji_code_output] = 1
            else: self.output_dict_attempted[emoji_code_output] += 1

    def f1(self, precision,recall):
        # this will return the f1 value for our evaluation
        if precision != 0.0 or recall != 0.0:
            return (2.0*precision*recall)/(precision+recall)
        return 0.0
    
    def evalulate(self):
        # this will return the f1 value for our evaluation
        num_emojis = len(self.truth_dict)
        f1_total = 0
        # This for loop is for precision, recall and F1 of each emoji.
        for emoji_code in self.truth_dict:
            gold_occurrences = self.truth_dict[emoji_code]
            if emoji_code in self.output_dict_attempted: attempted = self.output_dict_attempted[emoji_code]
            else: attempted = 0
            if emoji_code in self.output_dict_correct: correct = self.output_dict_correct[emoji_code]
            else: correct = 0
            if attempted != 0:
                precision = (correct*1.0) / attempted
                recall = (correct*1.0) / gold_occurrences
                f1_total += self.f1(precision, recall)
        return f1_total / (num_emojis*1.0)