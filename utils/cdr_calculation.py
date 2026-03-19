def calculate_cdr(disc_box, cup_box):

    disc_height = disc_box[3] - disc_box[1]
    cup_height = cup_box[3] - cup_box[1]

    if disc_height == 0:
        return 0

    return cup_height / disc_height