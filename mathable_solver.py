
import numpy as np
import os
import utils
from tqdm import tqdm
import cv2 as cv

class Solver():

    def __init__(self,data_set_id):

        self.data_set_id = data_set_id

        self.table_configuration = np.full((14,14),-1)
        self.table_configuration[6,6] = 1
        self.table_configuration[6,7] = 2
        self.table_configuration[7,6] = 3
        self.table_configuration[7,7] = 4

        self.ADD = 1
        self.SUB = 2
        self.MUL = 3
        self.DIV = 4
        self.TIMES_2 = 5
        self.TIMES_3 = 6

        self.table_constraints = np.array([
            [self.TIMES_3,0,0,0,0,0,self.TIMES_3,self.TIMES_3,0,0,0,0,0,self.TIMES_3],
            [0,self.TIMES_2,0,0,self.DIV,0,0,0,0,self.DIV,0,0,self.TIMES_2,0],
            [0,0,self.TIMES_2,0,0,self.SUB,0,0,self.SUB,0,0,self.TIMES_2,0,0],
            [0,0,0,self.TIMES_2,0,0,self.ADD,self.MUL,0,0,self.TIMES_2,0,0,0],
            [0,self.DIV,0,0,self.TIMES_2,0,self.MUL,self.ADD,0,self.TIMES_2,0,0,self.DIV,0],
            [0,0,self.SUB,0,0,0,0,0,0,0,0,self.SUB,0,0],
            [self.TIMES_3,0,0,self.MUL,self.ADD,0,0,0,0,self.MUL,self.ADD,0,0,self.TIMES_3],

            [self.TIMES_3,0,0,self.ADD,self.MUL,0,0,0,0,self.ADD,self.MUL,0,0,self.TIMES_3],
            [0,0,self.SUB,0,0,0,0,0,0,0,0,self.SUB,0,0],
            [0,self.DIV,0,0,self.TIMES_2,0,self.ADD,self.MUL,0,self.TIMES_2,0,0,self.DIV,0],
            [0,0,0,self.TIMES_2,0,0,self.MUL,self.ADD,0,0,self.TIMES_2,0,0,0],
            [0,0,self.TIMES_2,0,0,self.SUB,0,0,self.SUB,0,0,self.TIMES_2,0,0],
            [0,self.TIMES_2,0,0,self.DIV,0,0,0,0,self.DIV,0,0,self.TIMES_2,0],
            [self.TIMES_3,0,0,0,0,0,self.TIMES_3,self.TIMES_3,0,0,0,0,0,self.TIMES_3],

        ])


    def reset(self,output_dest_path = "./343_Gabroveanu_Razvan"):

        files = [x for x in os.listdir(output_dest_path) if x[0] == str(self.data_set_id)]

        for file in files:
            if os.path.exists(os.path.join(output_dest_path,file)):
                os.remove(os.path.join(output_dest_path,file))

        self.table_configuration = np.full((14,14),-1)
        self.table_configuration[6,6] = 1
        self.table_configuration[6,7] = 2
        self.table_configuration[7,6] = 3
        self.table_configuration[7,7] = 4


    def solve(self, src_dir ="./antrenare", output_dest_dir ="./343_Gabroveanu_Razvan"):

        self.reset(output_dest_dir)

        # extract the game tables and its relevant data
        game_tables = utils.extract_game(utils.get_image_paths(src_dir, self.data_set_id))
        relevant_game_tables = utils.extract_relevant_game(game_tables)

        assert len(game_tables) > 0, "Game tables not found"
        assert len(relevant_game_tables) > 0, "Relevant game tables not found"

        # prefix of where to store the predicted data
        output_file_prefix = "/" + str(self.data_set_id) + "_"

        turns = utils.process_turns(self.data_set_id,src_dir)
        current_turn_index = 0
        current_player,start_turn,end_turn = turns[current_turn_index]
        current_score = 0

        for i in tqdm(range(len(relevant_game_tables))):

            # get the current table, convert to gray and apply thresholding
            table = relevant_game_tables[i]
            table = cv.cvtColor(table, cv.COLOR_BGR2GRAY)
            _,table_binary = cv.threshold(table,65,255,cv.THRESH_BINARY_INV)

            # get the dict of predictions
            result_dict = self.match_numbers(table_binary,8)

            # forge the output path
            output_file_path = output_dest_dir + output_file_prefix

            if i in range(0,9):
                output_file_path = output_file_path + "0"

            output_file_path = output_file_path + str(i+1) + ".txt"

            # make sure to open and create the new file
            with open(output_file_path,"w") as file:
                # iterate through the predictions
                for indexes,predicted_number in result_dict.items():
                    row,col = indexes

                    # continue if the number was already predicted
                    if self.table_configuration[row,col] != -1:
                        continue

                    # write the output to its file
                    output = str(row+1) + chr(65 + col)+ " " + str(predicted_number)

                    file.write(output)

                    # mark the predicted number on the table
                    self.table_configuration[row,col] = predicted_number

                    current_score += self.calculate_score(row,col)

            if i+1 not in range(start_turn-1,end_turn-1):

                with open(output_dest_dir + "/" + str(self.data_set_id) + "_scores.txt", "a") as file:
                    file.write(current_player + " " + str(start_turn) + " " + str(current_score))
                    if i != 49:
                        file.write("\n")
                with open(output_dest_dir + "/" + str(self.data_set_id) + "_turns.txt", "a") as file:
                    file.write(current_player + " " + str(start_turn))
                    if i != 49:
                        file.write("\n")


                current_turn_index += 1
                current_player,start_turn,end_turn = turns[current_turn_index if current_turn_index < len(turns) else 0]
                current_score = 0


    def match_numbers(self,binary_image, offset):
        # get the vertical and horizontal line coordinates
        lines_vertical,lines_horizontal = utils.get_lines_coords()

        # result dict
        coords={}

        for i in range(len(lines_horizontal)-1):
            for j in range(len(lines_vertical)-1):

                # get the boundaries of the current patch
                y_min = lines_vertical[j][0][0] + offset
                y_max = lines_vertical[j + 1][1][0] - offset
                x_min = lines_horizontal[i][0][1] + offset
                x_max = lines_horizontal[i + 1][1][1] - offset

                patch = binary_image[x_min:x_max,y_min:y_max].copy()

                #ignore if the patches are on the middle
                if (i==6 or i==7) and (j==6 or j==7):
                    continue

                # check if there are any white pixels in the middle of the patch
                num_white_pixels = np.sum(binary_image[x_min+30:x_max-30,y_min+30:y_max-30].copy() == 255)

                # test if there are white pixels and the current patch was not already predicted
                if num_white_pixels > 0 and self.table_configuration[i,j] == -1:

                    # get the possible values that the chosen patch can take
                    possible_numbers = self.get_possible_values(i,j)

                    # get the predicted number of the patch and store it in the dictironary
                    predicted_number = self.classify_number(patch,possible_numbers)
                    coords[(i,j)] = predicted_number

        return coords

    def classify_number(self,patch,possible_numbers,templates_path = "./templates_cropped"):

        max_corr = -np.inf
        chosen_number = -2

        one_digit_numbers = [x for x in possible_numbers if x < 10]
        more_digits_numbers = [x for  x in possible_numbers if x >=10]

        # test the correlation on the possible_numbers
        for number in more_digits_numbers:

            path = templates_path+ "/" + str(number) + ".jpg"

            # skip if the current number is not a valid piece
            if not os.path.exists(path):
                continue

            template = cv.imread(path)
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

            corr = cv.matchTemplate(patch,template, cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)

            if corr > max_corr:
                max_corr = corr
                chosen_number = number

        if max_corr >= 0.75:
            return chosen_number

        for number in one_digit_numbers:

            path = templates_path+ "/" + str(number) + ".jpg"

            # skip if the current number is not a valid piece
            if not os.path.exists(path):
                continue

            template = cv.imread(path)
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

            corr = cv.matchTemplate(patch,template, cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)

            if corr > max_corr:
                max_corr = corr
                chosen_number = number



        if chosen_number != -2:
            return chosen_number

        # if there are no detections just test on all templates
        all_numbers = utils.get_mathable_pieces_numbers()

        for number in all_numbers:

            path = templates_path+ "/" + str(number) + ".jpg"

            template = cv.imread(path)
            template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

            corr = cv.matchTemplate(patch,template, cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)

            if corr > max_corr:
                max_corr = corr
                chosen_number = number


        return chosen_number



    def get_neighbour_cells_indexes(self, i, j):
        neighbours = []

        # UP
        if i > 1 and self.table_configuration[i-1,j] != -1 and self.table_configuration[i-2,j] != -1:
            neighbours.append([(i-1,j),(i-2,j)])
        #DOWN
        if i < 14 - 2 and self.table_configuration[i+1,j] != -1 and self.table_configuration[i+2,j] != -1:
            neighbours.append([(i+1,j),(i+2,j)])
        # LEFT
        if j > 1 and self.table_configuration[i,j-1] != -1 and self.table_configuration[i,j-2] != -1:
            neighbours.append([(i,j-1),(i,j-2)])
        # RIGHT
        if j < 14 - 2 and self.table_configuration[i,j+1] != -1 and self.table_configuration[i,j+2] != -1:
            neighbours.append([(i,j+1),(i,j+2)])

        return neighbours


    def get_possible_values(self,i,j):

        neighbours_indexes = self.get_neighbour_cells_indexes(i,j)

        possible_piece_values = []

        current_cell_constraint = self.table_constraints[i,j]

        for pair in neighbours_indexes:
            i1,j1 = pair[0]
            i2,j2 = pair[1]

            neigh1 = self.table_configuration[i1,j1]
            neigh2 = self.table_configuration[i2,j2]

            # Check constraints
            if current_cell_constraint == self.ADD:
                possible_piece_values.append(neigh2 + neigh1)
                continue

            if current_cell_constraint == self.MUL:
                possible_piece_values.append(neigh2 * neigh1)
                continue

            if current_cell_constraint == self.DIV:

                if neigh2 != 0 and neigh1 % neigh2 == 0 :
                    possible_piece_values.append(neigh1//neigh2)

                elif neigh1 != 0 and neigh2 % neigh1 == 0 :
                    possible_piece_values.append(neigh2//neigh1)

                continue

            if current_cell_constraint == self.SUB:
                possible_piece_values.append(abs(neigh2 - neigh1))
                continue

            # No constraints case
            possible_piece_values.append(neigh2 + neigh1)
            possible_piece_values.append(neigh2 * neigh1)
            possible_piece_values.append(abs(neigh2 - neigh1))

            if neigh2 != 0 and neigh1 % neigh2 == 0 :
                possible_piece_values.append(neigh1//neigh2)

            elif neigh1 != 0 and neigh2 % neigh1 == 0 :
                possible_piece_values.append(neigh2//neigh1)

        return list(set(possible_piece_values))


    def calculate_score(self,i,j):

        assert self.table_configuration[i,j] != -1, "A cell must have a value in order to calculate a score."

        neighbours_indexes = self.get_neighbour_cells_indexes(i,j)
        value = self.table_configuration[i,j]
        current_constraint = self.table_constraints[i,j]

        score = 0

        # Just addition score
        if current_constraint == self.ADD:

            for pair in neighbours_indexes:
                i1,j1 = pair[0]
                i2,j2 = pair[1]

                neigh1 = self.table_configuration[i1,j1]
                neigh2 = self.table_configuration[i2,j2]

                score += value if neigh1 + neigh2 == value else 0
            return score

        # Just multiply score
        if current_constraint == self.MUL:

             for pair in neighbours_indexes:
                i1,j1 = pair[0]
                i2,j2 = pair[1]

                neigh1 = self.table_configuration[i1,j1]
                neigh2 = self.table_configuration[i2,j2]

                score += value if neigh1 * neigh2 == value else 0
             return score

        # Just subtraction score
        if current_constraint == self.SUB:

            for pair in neighbours_indexes:
                i1,j1 = pair[0]
                i2,j2 = pair[1]

                neigh1 = self.table_configuration[i1,j1]
                neigh2 = self.table_configuration[i2,j2]

                score += value if abs(neigh1 - neigh2) == value else 0
            return score

        # Just division score
        if current_constraint == self.DIV:

            for pair in neighbours_indexes:
                i1,j1 = pair[0]
                i2,j2 = pair[1]

                neigh1 = self.table_configuration[i1,j1]
                neigh2 = self.table_configuration[i2,j2]

                if neigh2 != 0 and neigh1 % neigh2 == 0:
                    score += value if neigh1//neigh2 == value else 0

                elif neigh1 != 0 and neigh2 % neigh1 == 0:
                    score += value if neigh2//neigh1 == value else 0
            return score


        # No operator constraint, test them all

        for pair in neighbours_indexes:
                i1,j1 = pair[0]
                i2,j2 = pair[1]

                neigh1 = self.table_configuration[i1,j1]
                neigh2 = self.table_configuration[i2,j2]

                if neigh1 + neigh2 == value or neigh1 * neigh2 == value or abs(neigh1 - neigh2) == value:
                    score += value
                    continue

                if (neigh2 != 0 and neigh1 % neigh2 == 0 and neigh1//neigh2 == value) or (neigh1 != 0 and neigh2 % neigh1 == 0 and neigh2//neigh1==value):

                    score += value
                    continue

        # return the score and check for bonus constraints
        return score * 3 if current_constraint == self.TIMES_3 else score * 2 if current_constraint == self.TIMES_2 else score
