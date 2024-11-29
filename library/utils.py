
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def show_image_cv(title,image,fx=1.0,fy=1.0,output = True):
    """
    Displays an imagine using opencv

    :param title: str
    :param image: image to show
    :param fx: scale on x
    :param fy: scale on y
    :param output: if true, show the image
    :return: None
    """
    if not output:
        return

    image = cv.resize(image,(0,0),fx=fx,fy=fy)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def show_image_matplot(title, image,fx=1.0,fy=1.0,output = True):

    """
    Displays an image using matplotlib
    :param title: str
    :param image: image to show
    :param fx: scale on x
    :param fy: scale on y
    :param output: if true, show the image
    :return: None
    """
    if not output:
        return

    image_resized = cv.resize(image, (0, 0), fx=fx, fy=fy)

    image = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(title)
    plt.axis('off')  # Ascunde axele pentru a afișa doar imaginea
    plt.show()

def show_images_matplot(images,fx=1,fy=1,output = True,num_images = 1):

    num_images = min(num_images,len(images))

    for i in range(num_images):
        image = images[i]
        show_image_matplot("img "+str(i),image,fx,fy,output)


def show_images_cv(images, fx=1, fy=1, output=True, num_images=1):
    num_images = min(num_images, len(images))

    for i in range(num_images):
        image = images[i]
        show_image_cv("img " + str(i), image, fx, fy, output)

def get_image_paths(dir,index):
    """
    Returns all paths to jpg images that start with index
    :param dir: directory path of the images
    :param index: number of the image
    :return: [str]
    """
    image_paths = os.listdir(dir)
    image_paths = ["./" + dir + "/" + file for file in image_paths if file[0:2] == str(index) + "_" and file[-4:] == ".jpg"]
    return sorted(image_paths)


def get_outer_masked_image(img,output = False):
    """

    Returns a masked mathable image with the brown table removed

    :param img: Mathable image
    :param output: if true, show intermediate images
    :return: Masked mathable table
    """
    lower_hsv_bound = np.array([20,0,0])
    upper_hsv_bound = np.array([255,255,255])

    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv_image, lower_hsv_bound, upper_hsv_bound)
    show_image_cv("mask",mask,output = output)

    new_image = cv.bitwise_and(img, img, mask=mask)
    show_image_cv("ceva",new_image,output = output)

    return new_image



def get_game_countour(image,output = False):

    """

    :param image: Mathable image
    :param output: if true, show intermediate images
    :return: Cropped mathable game image
    """

    masked_image = get_outer_masked_image(image)
    masked_image_grey = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)

    show_image_cv("masked_image_grey",masked_image_grey,output = output,fx=0.3,fy=0.3)

    masked_image_grey = cv.medianBlur(masked_image_grey,3)
    show_image_cv('median_blur_image',masked_image_grey,output = output,fx=0.3,fy=0.3)

    contours, _ = cv.findContours(masked_image_grey,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for i in range(len(contours)):
            if(len(contours[i]) >3):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis = 1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left


    image_copy = cv.cvtColor(masked_image_grey.copy(),cv.COLOR_GRAY2BGR)
    cv.circle(image_copy,tuple(top_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),20,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),20,(0,0,255),-1)
    show_image_cv("detected corners",image_copy,output = output,fx=0.3,fy=0.3)


    width = 810
    height = 810

    puzzle = np.array([top_left,top_right,bottom_left,bottom_right],dtype=np.float32)
    destination = np.array([[0,0],[width,0],[0,height],[width,height]],dtype=np.float32)
    M = cv.getPerspectiveTransform(puzzle,destination)
    result = cv.warpPerspective(image,M,(width,height))

    return result


def extract_game(image_paths,size = 1000):
    """
    Extract the mathable game table from the images.
    :param image_paths: paths to the images
    :param size:
    :return: np.array[images]
    """
    images = []
    size = size if size <= len(image_paths) else len(image_paths)
    for i in tqdm(range(size)):
        img = cv.imread(image_paths[i])
        result = get_game_countour(img)
        images.append(result)
    return np.array(images)

def strip_margins(game_table_image, cell_size = 150, output = False):

    """
    Crop the irrelevant outer layer of a mathable game table
    :param game_table_image: cropped mathable image
    :param cell_size: the size in pixels of each game cell after cropping the outer layer
    :param output: if true, show intermediate images
    :return:  Cropped game table
    """

    game_table_image = game_table_image.copy()
    img_grey = cv.cvtColor(game_table_image, cv.COLOR_BGR2GRAY)
    BLACK = 0

    height,width,_ = game_table_image.shape

    # somewhere between 13% and 14% of the height and width of the image
    top_margin = 107
    bottom_margin = height - 102
    left_margin = 107
    right_margin = width - 105

    img_grey[:top_margin, :] = BLACK
    img_grey[bottom_margin:, :] = BLACK

    img_grey[:, :left_margin] = BLACK
    img_grey[:, right_margin:] = BLACK

    show_image_cv("ceva",img_grey,fx=1,fy=1,output = output)

    masked_image_grey = img_grey

    contours, _ = cv.findContours(masked_image_grey,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for i in range(len(contours)):
            if(len(contours[i]) >3):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis = 1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left


    image_copy = cv.cvtColor(masked_image_grey.copy(),cv.COLOR_GRAY2BGR)
    cv.circle(image_copy,tuple(top_left),2,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),2,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),2,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),2,(0,0,255),-1)
    show_image_cv("detected corners",image_copy,output = output,fx=1,fy=1)


    width = 14 * cell_size
    height = 14 * cell_size

    puzzle = np.array([top_left,top_right,bottom_left,bottom_right],dtype=np.float32)
    destination = np.array([[0,0],[width,0],[0,height],[width,height]],dtype=np.float32)
    M = cv.getPerspectiveTransform(puzzle,destination)
    result = cv.warpPerspective(game_table_image, M, (width, height))

    return result

def extract_relevant_game(game_table_images):

    """
    Crops the irrelevant outer layer of multiple mathable game tables
    :param game_table_images:
    :return: np.array[image]
    """

    new_images = []

    for image in tqdm(game_table_images):
        new_images.append(strip_margins(image))

    return np.array(new_images)


def get_lines_coords(cell_size = 150):

    """
    Returns two lists of the coordinates of the vertical and horizontal lines that will separate each mathable game cell
    :param cell_size: The cell size in pixels
    :return: ([coords vertical],[coords horizontal])
    """

    lines_horizontal = []
    for i in range(0,14*cell_size+1,cell_size):
        line = []
        line.append((0,i))
        line.append((14*cell_size-1,i))
        lines_horizontal.append(line)

    lines_vertical = []
    for i in range(0,14*cell_size+1,cell_size):
        line = []
        line.append((i,0))
        line.append((i,14*cell_size-1))
        lines_vertical.append(line)

    return lines_vertical,lines_horizontal

def draw_lines(images):
    """
    Draw the horizontal and vertical lines onto multiple images.
    :param images:
    :return: Copies of the original images, with lines drawn
    """
    new_images = images.copy()

    lines_vertical,lines_horizontal = get_lines_coords()

    GREEN = (0,255,0)
    RED = (0,0,255)

    for image in tqdm(new_images):
        for line in lines_vertical:
            cv.line(image,line[0],line[1],GREEN,2)

        for line in lines_horizontal:
            cv.line(image,line[0],line[1],RED,2)

    return new_images

def get_mathable_pieces_numbers():
    """
    Returns a list of each mathable piece number.
    :return: List of each piece number.
    """
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]


def store_template_numbers1(offset = 0,store_path = "./templates/sample1"):

    """
    Reads the first image witch contains a mathable table with all pieces and extracts those pieces as template images.
    :param offset: number of pixels to crop on each side
    :param store_path: where to store the template numbers
    :return: None
    """

    img = ["./imagini_auxiliare/03.jpg"]
    table = extract_game(img)
    game_table = extract_relevant_game(table)[0]

    lines_vertical,lines_horizontal = get_lines_coords()

    piece_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]

    index = 0

    for i in range(5,11):
        for j in range(4,12):

            if index == len(piece_values):
                return

            y_min = lines_vertical[j][0][0] + offset
            y_max = lines_vertical[j + 1][1][0] - offset
            x_min = lines_horizontal[i][0][1] + offset
            x_max = lines_horizontal[i + 1][1][1] - offset

            patch = game_table[x_min:x_max,y_min:y_max:].copy()
            cv.imwrite(store_path+"/" + str(piece_values[index])+ ".jpg", patch)
            index += 1



def store_template_numbers2(offset = 5,store_path = "./templates/sample2"):
    """
    Reads the second image witch contains a mathable table with all pieces and extracts those pieces as template images.
    :param offset: number of pixels to crop on each side
    :param store_path: where to store the template numbers
    :return: None
    """

    img = ["./imagini_auxiliare/04.jpg"]
    table = extract_game(img)
    game_table = extract_relevant_game(table)[0]

    lines_vertical,lines_horizontal = get_lines_coords()

    piece_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 80, 81, 90]

    index = 0

    for i in range(0,len(lines_horizontal)-1,2):
        for j in range(0,len(lines_vertical)-1,2):

            if index == len(piece_values):
                return

            y_min = lines_vertical[j][0][0] + offset
            y_max = lines_vertical[j + 1][1][0] - offset
            x_min = lines_horizontal[i][0][1] + offset
            x_max = lines_horizontal[i + 1][1][1] - offset

            patch = game_table[x_min:x_max,y_min:y_max:].copy()
            cv.imwrite(store_path+ "/" + str(piece_values[index])+ ".jpg", patch)
            index += 1


def store_binary_templates(thresh = 100,source_path = "./templates/sample2",destination_path = "./templates_binary/sample2"):
    """

    Reads templates and applies a threshold then stores them.

    :param thresh: threshold for the template numbers
    :param source_path: path to get the template images
    :param destination_path: path to store the template images
    :return:
    """
    paths = sorted(os.listdir(source_path))

    for path in paths:
        img = cv.imread(os.path.join(source_path,path))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, img = cv.threshold(img,thresh,255,cv.THRESH_BINARY_INV)
        cv.imwrite(os.path.join(destination_path,path), img)

def crop_and_store_templates(templates_path = "./templates_binary/sample2",store_path = "./templates_cropped"):

    """
    Reads binary templates from templates path, cropps them and stores them to store path.
    :param templates_path:
    :param store_path:
    :return: None
    """

    files = os.listdir(templates_path)

    for file in tqdm(files):
        path = templates_path + "/" + file
        template = cv.imread(path)
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        height,width= template.shape
        x_min = -1
        x_max = -1
        y_min = -1
        y_max = -1

        for i in range(0,height):
            if y_min != -1:
                break
            for j in range(0,width):
                if template[i,j] == 255:
                    y_min = i
                    break

        for j in range(0,width):
            if x_min != -1:
                break
            for i in range(0,height):
                if template[i,j] == 255:
                    x_min = j
                    break

        for i in range(height-1,0,-1):
            if y_max != -1:
                break
            for j in range(0,width):
                if template[i,j] == 255:
                    y_max = i
                    break

        for j in range(width-1,0,-1):
            if x_max != -1:
                break
            for i in range(0,height):
                if template[i,j] == 255:
                    x_max = j
                    break

        cropped_template = template[y_min:y_max, x_min:x_max]

        cv.imwrite(store_path+"/"+file,cropped_template)

def select_and_save_roi(image_path, save_path):

    """
    Reads and image from image_path, selects with ROI an area and saves it to save_path.
    :param image_path:
    :param save_path:
    :return: None
    """

    image = cv.imread(image_path)

    roi = cv.selectROI("Select ROI", image, showCrosshair=True)
    cv.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi)
    selected_area = image[y:y+h, x:x+w]

    cv.imwrite(save_path, selected_area)

def rewrite_all(src_path = "./templates_handmade/new",dest_path = "./templates_cropped"):

    """
    Copyes and rewrites if there are name collisions the giles from source to destination.
    :param src_path:
    :param dest_path:
    :return: None
    """

    files = os.listdir(src_path)

    for file in files:
        old_path = src_path + "/" + file
        new_path = dest_path + "/" + file
        image = cv.imread(old_path)
        cv.imwrite(new_path,image)


def move_train_data(train_number,src_path = "./antrenare",dest_path = "./train"):

    """
    Moves all training data labeled with train_number from src_path to dest_path.
    :param train_number:
    :param src_path:
    :param dest_path:
    :return: None
    """

    img_prefix = str(train_number)+ "_"

    for i in range(9):
        img_name = img_prefix + "0" +str(i+1) + ".jpg"
        img_path = src_path + "/" + img_name
        img = cv.imread(img_path)
        cv.imwrite(dest_path +"/" + img_name, img)

    for i in range(9,50):
        img_path = src_path + "/" + img_prefix + str(i+1) + ".jpg"
        img_name = img_prefix +str(i+1) + ".jpg"
        img = cv.imread(img_path)
        cv.imwrite(dest_path +"/" + img_name, img)

def process_turns(data_id,src_dir = "./antrenare"):
    """
    Parses the turns file and returns an array of tuples with the current player, the start turn and the end turn.
    :param data_id: value between 1 and 4
    :param src_dir: source directory of where the file data_id_turns.txt are located
    :return: [(player,current turn,end turn)]
    """
    result = []
    path = src_dir + "/" + str(data_id) + "_turns.txt"

    with open(path) as file:
        current_line = file.readline()
        next_line = file.readline()

        while next_line:
            current_player = current_line.strip().split()[0]
            current_turn = current_line.strip().split()[1]
            end_turn = next_line.strip().split()[1]
            result.append((current_player, int(current_turn), int(end_turn)))

            current_line = next_line
            next_line = file.readline()

        current_player = current_line.strip().split()[0]
        current_turn = current_line.strip().split()[1]
        result.append((current_player, int(current_turn),51))

    return result